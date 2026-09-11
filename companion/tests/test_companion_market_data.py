"""Original synthetic HTTP fixtures; no venue request or real payload is used."""

import base64
import hashlib
import io
import json
from datetime import timedelta
from email.message import Message
from http.client import HTTPResponse, IncompleteRead
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from uuid import uuid4

from prediction_market_lab.market_data import (
    CLOB, GAMMA, KALSHI, DisallowedRequest, PublicGetTransport, PublicResponse,
    RequestPolicy, SchemaError, SnapshotClient, SnapshotError, _NoRedirect,
    main, snapshot_output_path, validate_public_url, write_snapshot,
)
from prediction_market_lab.validation import parse_utc

NOW = parse_utc("2025-07-01T12:00:00Z")


def kalshi_market(ticker="TOY-EVENT-YES"):
    return {
        "ticker": ticker, "event_ticker": "TOY-EVENT", "market_type": "binary",
        "status": "active", "rules_primary": "Entirely invented test contract.", "rules_secondary": "",
        "yes_bid_dollars": "0.4210", "yes_ask_dollars": "0.4350", "volume_fp": "12.75",
        "price_ranges": [{"start": "0.01", "end": "0.99", "step": "0.001"}],
        "future_unknown_field": {"original": True},
    }


def kalshi_book():
    return {"orderbook_fp": {"yes_dollars": [["0.4210", "2.75"], ["0.4010", "3.25"]],
                             "no_dollars": [["0.5550", "4.50"]]}}


def gamma_market(identifier="101"):
    return {"id": identifier, "question": "Original synthetic question", "conditionId": "0xtoy",
            "clobTokenIds": '["123", "456"]', "outcomes": '["YES", "NO"]',
            "outcomePrices": '["0.43", "0.57"]', "closed": False,
            "denominationToken": "UNVERIFIED_NATIVE_COLLATERAL", "custom_nested": {"keep": "unchanged"}}


def clob_book():
    return {"market": "0xtoy", "asset_id": "123", "timestamp": "1751371200000", "hash": "toy-state-hash",
            "bids": [{"price": "0.421", "size": "2.75"}],
            "asks": [{"price": "0.435", "size": "4.25"}],
            "min_order_size": "1.5", "tick_size": "0.001", "neg_risk": False,
            "last_trade_price": "0.429", "unknown_native_field": "retain"}


class FakeTime:
    def __init__(self):
        self.seconds = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.seconds

    def clock(self):
        return NOW + timedelta(seconds=self.seconds)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.seconds += seconds


class FakeTransport:
    def __init__(self, replies, clock):
        self.replies, self.clock, self.calls = list(replies), clock, []

    def __call__(self, url, *, timeout, max_bytes):
        self.calls.append({"url": url, "timeout": timeout, "max_bytes": max_bytes})
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        payload = reply.get("payload", {})
        raw = reply.get("raw", json.dumps(payload, separators=(", ", ": ")).encode("utf-8"))
        headers = {"Content-Type": "application/json", **reply.get("headers", {})}
        return PublicResponse(reply.get("status", 200), headers, raw, self.clock(),
                              reply.get("final_url", url))


def make_client(replies, *, retries=0):
    clock = FakeTime()
    transport = FakeTransport(replies, clock.clock)
    client = SnapshotClient(acknowledge_terms=True, policy=RequestPolicy(retries=retries),
                            transport=transport, clock=clock.clock, monotonic=clock.monotonic,
                            sleep=clock.sleep)
    return client, transport, clock


class PermissionAndURLTests(unittest.TestCase):
    def test_permission_required_even_with_injected_transport(self):
        transport = FakeTransport([], lambda: NOW)
        for acknowledgement in (False, None, "True", 1):
            with self.subTest(value=acknowledgement), self.assertRaises(DisallowedRequest):
                SnapshotClient(acknowledge_terms=acknowledgement, transport=transport)
        self.assertEqual(transport.calls, [])
        with patch("prediction_market_lab.market_data.build_opener") as build:
            with self.assertRaises(DisallowedRequest):
                PublicGetTransport()
            build.assert_not_called()

    def test_only_four_explicit_public_routes(self):
        urls = {
            KALSHI + "/markets?limit=20&status=open": "kalshi-markets",
            KALSHI + "/markets/TOY-YES/orderbook?depth=20": "kalshi-book",
            GAMMA + "/markets?limit=20&offset=0&closed=false&order=id&ascending=true": "polymarket-markets",
            CLOB + "/book?token_id=123": "polymarket-book",
        }
        for url, kind in urls.items():
            self.assertEqual(validate_public_url(url), kind)

    def test_disallowed_hosts_paths_credentials_and_query_injections(self):
        urls = [
            "http://clob.polymarket.com/book?token_id=123",
            "https://localhost/book?token_id=123",
            "https://clob.polymarket.com.evil.test/book?token_id=123",
            "https://user:password@clob.polymarket.com/book?token_id=123",
            "https://clob.polymarket.com:443/book?token_id=123",
            CLOB + "/book?token_id=123#fragment",
            CLOB + "/book?token_id=123&api_key=not-allowed",
            CLOB + "/book?token_id=123&token_id=456",
            CLOB + "/book?token_id=123%0d%0aX-Foo:bad",
            CLOB + "/book?token_id=../order",
            CLOB + "/order",
            CLOB + "/books",
            KALSHI + "/portfolio/orders",
            KALSHI + "/markets/TOY%2fportfolio/orderbook?depth=20",
            KALSHI + "/markets/../orderbook?depth=20",
            KALSHI + "/historical/markets?limit=20",
            KALSHI + "/markets?limit=1001",
            KALSHI + "/markets?limit=20&cursor=%0a",
            KALSHI + "/markets/TOY/orderbook?depth=0",
            GAMMA + "/markets?limit=20&offset=0&closed=false&order=volume&ascending=true",
            "https://[broken",
        ]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(SnapshotError):
                validate_public_url(url)

    def test_input_validation_precedes_transport(self):
        client, transport, _ = make_client([])
        calls = [
            lambda: client.kalshi_markets(page_size=0),
            lambda: client.kalshi_markets(max_pages=11),
            lambda: client.kalshi_markets(status="all"),
            lambda: client.kalshi_markets(series_ticker="../orders"),
            lambda: client.kalshi_book("../orders"),
            lambda: client.kalshi_book("TOY", depth=0),
            lambda: client.polymarket_markets(closed="false"),
            lambda: client.polymarket_book("123&token_id=456"),
        ]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(SnapshotError):
                call()
        self.assertEqual(transport.calls, [])

    def test_policy_is_bounded(self):
        for kwargs in ({"timeout_seconds": float("nan")}, {"timeout_seconds": 0},
                       {"timeout_seconds": 31}, {"retries": 3}, {"retries": True},
                       {"minimum_interval_seconds": 0}, {"max_response_bytes": 8 * 1024 * 1024}):
            with self.subTest(kwargs=kwargs), self.assertRaises(SnapshotError):
                RequestPolicy(**kwargs)


class SchemaAndProvenanceTests(unittest.TestCase):
    def test_kalshi_native_book_strings_fractional_size_and_original_order(self):
        payload = kalshi_book()
        client, transport, _ = make_client([{"payload": payload}])
        result = client.kalshi_book("TOY-YES")
        page = result["pages"][0]
        self.assertEqual(page["payload"], payload)
        self.assertEqual(page["payload"]["orderbook_fp"]["yes_dollars"][0], ["0.4210", "2.75"])
        self.assertIn("depth=20", transport.calls[0]["url"])
        self.assertTrue(result["book_depth_limit_may_omit_levels"])
        self.assertFalse(result["complete_book_history"])

    def test_clob_fields_no_collateral_or_fee_coercion(self):
        payload = clob_book()
        client, _, _ = make_client([{"payload": payload}])
        result = client.polymarket_book("123")
        self.assertEqual(result["pages"][0]["payload"], payload)
        self.assertEqual(result["pages"][0]["validated_summary"]["venue_timestamp_native"], "1751371200000")
        self.assertEqual(result["collateral_fees_and_payouts"], "VENUE_NATIVE_FIELDS_NOT_COERCED_OR_VERIFIED")
        self.assertIn("not the US", make_client([{"payload": []}])[0].polymarket_markets()["product"])

    def test_gamma_encoded_fields_are_preserved_not_rewritten(self):
        payload = [gamma_market()]
        client, _, _ = make_client([{"payload": payload}])
        page = client.polymarket_markets()["pages"][0]
        self.assertEqual(page["payload"], payload)
        self.assertIsInstance(page["payload"][0]["clobTokenIds"], str)

    def test_raw_hash_exact_bytes_and_unknown_native_fields(self):
        payload = {"markets": [kalshi_market()], "cursor": None, "new_envelope_field": "keep"}
        body = json.dumps(payload, indent=3).encode("utf-8") + b"\n"
        client, _, _ = make_client([{"raw": body, "headers": {
            "ETag": '"native-etag"', "Set-Cookie": "do-not-record", "Authorization": "do-not-record"}}])
        result = client.kalshi_markets()
        page = result["pages"][0]
        self.assertEqual(base64.b64decode(page["raw_body_base64"]), body)
        self.assertEqual(page["raw_sha256"], hashlib.sha256(body).hexdigest())
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.assertEqual(page["payload_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(page["payload"], payload)
        self.assertEqual(page["response_headers"]["etag"], '"native-etag"')
        self.assertNotIn("set-cookie", page["response_headers"])
        self.assertNotIn("authorization", page["response_headers"])
        self.assertEqual(page["received_at_utc"], NOW.isoformat())
        self.assertIsNone(page["publication_time"])
        self.assertEqual(result["publication_history"], "UNKNOWN")
        self.assertEqual(result["snapshot_scope"], "CURRENT_ONLY")
        self.assertFalse(result["historical_point_in_time"])
        self.assertFalse(result["eligibility_or_redistribution_rights_verified"])
        self.assertFalse(result["live_edge_claim"])
        self.assertEqual(result["schema_reference"]["documentation_checked_on"], "2026-09-10")
        self.assertEqual(result["attempts"][0]["method"], "GET")

    def test_empty_books_are_valid_not_synthetic_liquidity(self):
        payload = kalshi_book()
        payload["orderbook_fp"] = {"yes_dollars": [], "no_dollars": []}
        client, _, _ = make_client([{"payload": payload}])
        self.assertEqual(client.kalshi_book("TOY")["pages"][0]["payload"], payload)

    def test_legacy_or_malformed_kalshi_book_rejected(self):
        bad_payloads = [
            {"orderbook": {"yes": [[40, 10]], "no": [[50, 10]]}},
            {"orderbook_fp": {"yes_dollars": [[0.4, "1.0"]], "no_dollars": []}},
            {"orderbook_fp": {"yes_dollars": [["0.4", "NaN"]], "no_dollars": []}},
            {"orderbook_fp": {"yes_dollars": [["1.4", "1.0"]], "no_dollars": []}},
            {"orderbook_fp": {"yes_dollars": [["0.4", "-1.0"]], "no_dollars": []}},
            {"orderbook_fp": {"yes_dollars": None, "no_dollars": []}},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(SchemaError):
                make_client([{"payload": payload}])[0].kalshi_book("TOY")

    def test_clob_mismatch_and_missing_native_fields_rejected(self):
        for key, value in (("asset_id", "456"), ("timestamp", 123), ("neg_risk", "false"),
                           ("tick_size", 0.01), ("bids", [{"price": "0.4", "size": "-2"}])):
            payload = clob_book()
            payload[key] = value
            with self.subTest(field=key), self.assertRaises(SchemaError):
                make_client([{"payload": payload}])[0].polymarket_book("123")

    def test_listing_envelopes_limits_and_encoded_arrays_validated(self):
        bad = gamma_market()
        bad["clobTokenIds"] = '["123", 456]'
        for kind, payload in (("gamma", {"items": []}), ("gamma", [bad]),
                              ("kalshi", {"markets": []}),
                              ("kalshi", {"markets": [{"ticker": "x"}], "cursor": ""})):
            with self.subTest(kind=kind, payload=payload), self.assertRaises(SchemaError):
                client = make_client([{"payload": payload}])[0]
                client.polymarket_markets() if kind == "gamma" else client.kalshi_markets()
        with self.assertRaises(SchemaError):
            make_client([{"payload": [gamma_market("1"), gamma_market("2")]}])[0].polymarket_markets(page_size=1)

    def test_invalid_json_utf8_duplicate_keys_and_nonfinite_values_rejected(self):
        for raw in (b"not-json", b"\xff", b'{"markets": [], "markets": [], "cursor": ""}',
                    b'{"markets": [], "cursor": "", "x": NaN}',
                    b'{"markets": [], "cursor": "", "x": 1e999}',
                    b'{"markets": [], "cursor": "", "x": "\\ud800"}'):
            with self.subTest(raw=raw), self.assertRaises(SchemaError):
                make_client([{"raw": raw}])[0].kalshi_markets()

    def test_html_content_type_oversize_and_compression_rejected(self):
        for reply in ({"headers": {"Content-Type": "text/html"}},
                      {"raw": b" " * (2 * 1024 * 1024 + 1)},
                      {"headers": {"Content-Encoding": "gzip"}}):
            with self.subTest(reply=list(reply)), self.assertRaises(SnapshotError):
                make_client([reply])[0].kalshi_markets()


class PaginationAndFailureTests(unittest.TestCase):
    def test_kalshi_bounded_cursor_truncation_and_encoded_cursor(self):
        cursor = "opaque+/= &not_a_url"
        client, transport, clock = make_client([
            {"payload": {"markets": [kalshi_market("A")], "cursor": cursor}},
            {"payload": {"markets": [kalshi_market("B")], "cursor": "more"}},
        ])
        result = client.kalshi_markets(page_size=1, max_pages=2)
        self.assertTrue(result["pagination"]["truncated"])
        self.assertEqual(result["pagination"]["next_cursor"], "more")
        self.assertEqual(result["pagination"]["records_received"], 2)
        self.assertIn("cursor=opaque%2B%2F%3D+%26not_a_url", transport.calls[1]["url"])
        self.assertGreaterEqual(clock.seconds, 1)

    def test_kalshi_empty_or_null_cursor_exhaustion(self):
        for cursor in ("", None):
            client, transport, _ = make_client([{"payload": {"markets": [], "cursor": cursor}}])
            result = client.kalshi_markets(max_pages=10)
            self.assertFalse(result["pagination"]["truncated"])
            self.assertEqual(len(transport.calls), 1)

    def test_repeated_cursor_stops_without_loop(self):
        client, transport, _ = make_client([
            {"payload": {"markets": [kalshi_market("A")], "cursor": "repeat"}},
            {"payload": {"markets": [kalshi_market("B")], "cursor": "repeat"}},
        ])
        with self.assertRaisesRegex(SchemaError, "repeated") as caught:
            client.kalshi_markets(max_pages=10)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(len(caught.exception.completed_pages), 2)

    def test_gamma_offset_pagination_and_duplicate_records_retained(self):
        client, transport, _ = make_client([
            {"payload": [gamma_market("1"), gamma_market("2")]},
            {"payload": [gamma_market("2")]},
        ])
        result = client.polymarket_markets(page_size=2, max_pages=3)
        self.assertEqual(result["pagination"]["duplicate_identifiers_preserved"], ["2"])
        self.assertEqual(result["pagination"]["records_received"], 3)
        self.assertFalse(result["pagination"]["truncated"])
        self.assertIn("offset=2", transport.calls[1]["url"])
        self.assertEqual(result["pages"][1]["payload"][0]["id"], "2")

    def test_gamma_full_final_page_is_possibly_truncated_not_falsely_complete(self):
        client, _, _ = make_client([{"payload": [gamma_market("1")]}])
        result = client.polymarket_markets(page_size=1)
        self.assertTrue(result["pagination"]["truncated"])
        self.assertEqual(result["pagination"]["next_offset"], 1)

    def test_timeout_propagates_with_partial_page_metadata(self):
        client, transport, _ = make_client([
            {"payload": {"markets": [kalshi_market()], "cursor": "next"}},
            TimeoutError("synthetic timeout"),
        ])
        with self.assertRaisesRegex(SnapshotError, "TimeoutError") as caught:
            client.kalshi_markets(max_pages=2)
        self.assertEqual(len(caught.exception.completed_pages), 1)
        self.assertEqual(len(caught.exception.attempts), 2)
        self.assertEqual(len(transport.calls), 2)

    def test_rate_limit_retry_is_bounded_logged_and_honors_retry_after(self):
        client, transport, clock = make_client([
            {"status": 429, "headers": {"Retry-After": "3"}},
            {"payload": {"markets": [], "cursor": ""}},
        ], retries=1)
        result = client.kalshi_markets()
        self.assertEqual(len(transport.calls), 2)
        self.assertGreaterEqual(clock.seconds, 3)
        self.assertEqual(result["attempts"][0]["retry_delay_seconds"], 3)
        self.assertEqual(result["attempts"][0]["http_status"], 429)

    def test_incomplete_read_retries_preserve_prior_pages_and_attempts(self):
        client, transport, _ = make_client([
            {"payload": {"markets": [kalshi_market()], "cursor": "next"}},
            IncompleteRead(b"{", 15),
            IncompleteRead(b"{", 15),
        ], retries=1)
        with self.assertRaisesRegex(SnapshotError, "IncompleteRead") as caught:
            client.kalshi_markets(max_pages=2)
        self.assertEqual(len(caught.exception.completed_pages), 1)
        self.assertEqual(len(caught.exception.attempts), 3)
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(
            [attempt.get("error") for attempt in caught.exception.attempts],
            [None, "IncompleteRead", "IncompleteRead"],
        )

        client, _, _ = make_client([
            IncompleteRead(b"{", 15), {"payload": {"markets": [], "cursor": ""}},
        ], retries=1)
        result = client.kalshi_markets()
        self.assertEqual(len(result["pages"]), 1)
        self.assertEqual(result["attempts"][0]["error"], "IncompleteRead")

    def test_http_date_retry_after(self):
        client, _, clock = make_client([
            {"status": 503, "headers": {"Retry-After": "Tue, 01 Jul 2025 12:00:04 GMT"}},
            {"payload": []},
        ], retries=1)
        client.polymarket_markets()
        self.assertGreaterEqual(clock.seconds, 4)

    def test_excessive_or_invalid_retry_after_never_retries_early(self):
        for header in ("90", "not-a-date"):
            client, transport, _ = make_client([{"status": 429, "headers": {"Retry-After": header}}], retries=2)
            with self.subTest(header=header), self.assertRaises(SnapshotError):
                client.kalshi_markets()
            self.assertEqual(len(transport.calls), 1)

    def test_access_errors_never_retry_or_switch_to_authentication(self):
        for status in (301, 302, 307, 308, 400, 401, 403, 404, 451):
            client, transport, _ = make_client([{"status": status}], retries=2)
            with self.subTest(status=status), self.assertRaisesRegex(SnapshotError, f"HTTP {status}"):
                client.kalshi_markets()
            self.assertEqual(len(transport.calls), 1)

    def test_transient_and_network_retries_still_fail_explicitly_at_cap(self):
        for replies in ([{"status": 503}, {"status": 503}],
                        [URLError("synthetic failure"), URLError("synthetic failure")]):
            client, transport, _ = make_client(replies, retries=1)
            with self.assertRaises(SnapshotError):
                client.kalshi_markets()
            self.assertEqual(len(transport.calls), 2)

    def test_mismatched_final_url_rejected_even_for_custom_transport(self):
        client, transport, _ = make_client([{"payload": clob_book(), "final_url": CLOB + "/order"}])
        with self.assertRaises(DisallowedRequest):
            client.polymarket_book("123")
        self.assertEqual(len(transport.calls), 1)


class FakeHTTPResponse:
    def __init__(self, body=b"{}", *, headers=None, url=CLOB + "/book?token_id=123", chunks=None):
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.status, self.url, self.closed = 200, url, False
        self.data, self.chunks = io.BytesIO(body), chunks

    def geturl(self):
        return self.url

    def read1(self, limit):
        if self.chunks is not None:
            return self.chunks.pop(0) if self.chunks else b""
        return self.data.read(limit)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TransportAndCLITests(unittest.TestCase):
    def test_actual_chunked_read_failure_uses_structured_client_error(self):
        raw = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
               b"Transfer-Encoding: chunked\r\n\r\n10\r\n{")

        class MemorySocket:
            def makefile(self, *args, **kwargs):
                return io.BytesIO(raw)

        response = HTTPResponse(MemorySocket())
        response.begin()
        response.url = CLOB + "/book?token_id=123"
        client = SnapshotClient(
            acknowledge_terms=True, policy=RequestPolicy(retries=0), sleep=lambda _: None,
        )
        with patch("prediction_market_lab.market_data.build_opener") as build:
            build.return_value.open.return_value = response
            with self.assertRaisesRegex(SnapshotError, "IncompleteRead") as caught:
                client.polymarket_book("123")
        self.assertEqual(len(caught.exception.attempts), 1)
        self.assertIsInstance(caught.exception.__cause__, IncompleteRead)
        self.assertTrue(response.closed)

    def test_real_transport_constructs_only_fixed_get_headers_and_no_proxy(self):
        response = FakeHTTPResponse()
        with patch("prediction_market_lab.market_data.build_opener") as build:
            build.return_value.open.return_value = response
            result = PublicGetTransport(acknowledge_terms=True)(CLOB + "/book?token_id=123", timeout=2, max_bytes=1024)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.data)
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(set(headers), {"accept", "accept-encoding", "user-agent"})
        self.assertEqual(headers["accept-encoding"], "identity")
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertIsInstance(build.call_args.args[1], _NoRedirect)
        self.assertEqual(result.body, b"{}")
        self.assertTrue(response.closed)

    def test_transport_rejects_url_before_opener(self):
        with patch("prediction_market_lab.market_data.build_opener") as build:
            with self.assertRaises(DisallowedRequest):
                PublicGetTransport(acknowledge_terms=True)(CLOB + "/order", timeout=1, max_bytes=1024)
            build.assert_not_called()
        for timeout, max_bytes in ((1000, 1024), (2, 1024 * 1024 * 100), (2, -1)):
            with self.subTest(timeout=timeout, max_bytes=max_bytes), \
                    patch("prediction_market_lab.market_data.build_opener") as build:
                with self.assertRaises(SnapshotError):
                    PublicGetTransport(acknowledge_terms=True)(
                        CLOB + "/book?token_id=123", timeout=timeout, max_bytes=max_bytes)
                build.assert_not_called()

    def test_redirect_handler_never_constructs_a_followup_request(self):
        response = FakeHTTPResponse()
        for target in (CLOB + "/book?token_id=456", "https://evil.test/collect"):
            with self.subTest(target=target), self.assertRaises(DisallowedRequest):
                _NoRedirect().redirect_request(None, response, 302, "redirect", {}, target)
        self.assertTrue(response.closed)

    def test_response_size_cap_applies_with_and_without_content_length(self):
        responses = [FakeHTTPResponse(headers={"Content-Length": "2048"}),
                     FakeHTTPResponse(b"x" * 1025), FakeHTTPResponse(headers={"Content-Length": "-1"}),
                     FakeHTTPResponse(b"{}", headers={"Content-Length": "3"}),
                     FakeHTTPResponse(headers={"Content-Encoding": "gzip"})]
        for response in responses:
            with self.subTest(headers=response.headers), patch("prediction_market_lab.market_data.build_opener") as build:
                build.return_value.open.return_value = response
                with self.assertRaises(SnapshotError):
                    PublicGetTransport(acknowledge_terms=True)(CLOB + "/book?token_id=123", timeout=2, max_bytes=1024)
                self.assertTrue(response.closed)

    def test_socket_timeout_and_between_chunk_deadline(self):
        with patch("prediction_market_lab.market_data.build_opener") as build:
            build.return_value.open.side_effect = TimeoutError("synthetic")
            with self.assertRaises(TimeoutError):
                PublicGetTransport(acknowledge_terms=True)(CLOB + "/book?token_id=123", timeout=2, max_bytes=1024)
            self.assertEqual(build.return_value.open.call_args.kwargs["timeout"], 2)
        response = FakeHTTPResponse(chunks=[b"{", b"}"])
        with patch("prediction_market_lab.market_data.build_opener") as build, \
                patch("prediction_market_lab.market_data.time.monotonic", side_effect=[0, 1, 3]):
            build.return_value.open.return_value = response
            with self.assertRaises(TimeoutError):
                PublicGetTransport(acknowledge_terms=True)(CLOB + "/book?token_id=123", timeout=2, max_bytes=1024)

    def test_http_error_body_not_read_and_response_closed(self):
        headers = Message()
        headers["Retry-After"] = "2"
        body = io.BytesIO(b"not stored")
        error = HTTPError(CLOB + "/book?token_id=123", 429, "rate", headers, body)
        with patch("prediction_market_lab.market_data.build_opener") as build:
            build.return_value.open.side_effect = error
            result = PublicGetTransport(acknowledge_terms=True)(CLOB + "/book?token_id=123", timeout=2, max_bytes=1024)
        self.assertEqual(result.status, 429)
        self.assertEqual(result.body, b"")
        self.assertTrue(body.closed)

    def test_help_no_subcommand_and_missing_acknowledgement_are_offline(self):
        for args, code in ((["--help"], 0), (["kalshi-book", "--help"], 0), ([], 2),
                           (["kalshi-markets", "--output", "tests\\unused.snapshot.json"], 2)):
            with self.subTest(args=args), patch("prediction_market_lab.market_data.PublicGetTransport.__call__") as transport, \
                    patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(args)
                self.assertEqual(caught.exception.code, code)
                transport.assert_not_called()

    def test_bad_output_path_and_bad_identifier_do_not_acquire(self):
        for args in (
            ["kalshi-markets", "--acknowledge-terms", "--output", "tests\\not-a-snapshot.json"],
            ["kalshi-book", "--acknowledge-terms", "--ticker", "../bad", "--output", "tests\\unused.snapshot.json"],
        ):
            with patch("prediction_market_lab.market_data.PublicGetTransport.__call__") as transport, \
                    patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(main(args), 1)
                transport.assert_not_called()
        with self.assertRaises(SnapshotError):
            snapshot_output_path(Path("..") / "outside.snapshot.json")

    def test_output_is_exclusive_local_and_hashed(self):
        output = Path("tests") / f"synthetic-{uuid4().hex}.snapshot.json"
        self.addCleanup(output.unlink, missing_ok=True)
        fixture = {"classification": "SYNTHETIC TEST FIXTURE", "value": "0.4210"}
        checksum = write_snapshot(fixture, output)
        self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), checksum)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), fixture)
        with self.assertRaises(SnapshotError):
            write_snapshot(fixture, output)

    def test_cli_mocked_success_and_failure_never_fallback(self):
        output = Path("tests") / f"synthetic-{uuid4().hex}.snapshot.json"
        self.addCleanup(output.unlink, missing_ok=True)
        client, _, _ = make_client([{"payload": []}])
        with patch("prediction_market_lab.market_data.SnapshotClient", return_value=client), \
                patch("sys.stdout", new=io.StringIO()) as stdout:
            self.assertEqual(main(["polymarket-markets", "--acknowledge-terms", "--output", str(output)]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "CURRENT_SNAPSHOT_SAVED")
        output.unlink()
        client, _, _ = make_client([{"status": 403}])
        with patch("prediction_market_lab.market_data.SnapshotClient", return_value=client), \
                patch("sys.stderr", new=io.StringIO()) as stderr:
            self.assertEqual(main(["polymarket-markets", "--acknowledge-terms", "--output", str(output)]), 1)
            result = json.loads(stderr.getvalue())
            self.assertIs(result["synthetic_fallback"], False)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
