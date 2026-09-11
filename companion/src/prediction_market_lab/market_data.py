"""Opt-in public GET snapshots, isolated from paper accounting and trading.

Official endpoint/schema pages were checked 2026-09-10. Current responses are
not historical vintages. Public access is not permission to acquire/share data.
"""

import argparse
import base64
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .validation import utc

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CHECKED_ON = "2026-09-10"
TICKER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
TOKEN_ID = re.compile(r"(?:[0-9]{1,100}|0x[0-9a-fA-F]{1,128})")
DECIMAL_STRING = re.compile(r"[0-9]+(?:\.[0-9]+)?")
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
SAFE_HEADERS = frozenset({"content-type", "content-length", "content-encoding",
                          "date", "etag", "last-modified", "retry-after"})
REFERENCES = {
    "kalshi-markets": {
        "venue": "Kalshi",
        "product": "Kalshi public event-market metadata",
        "documented_api_version": "3.30.0",
        "urls": [
            "https://docs.kalshi.com/api-reference/market/get-markets.md",
            "https://docs.kalshi.com/getting_started/quick_start_market_data.md",
            "https://docs.kalshi.com/getting_started/pagination.md",
        ],
    },
    "kalshi-book": {
        "venue": "Kalshi",
        "product": "Kalshi public YES/NO bid snapshot; native fixed-point quantities",
        "documented_api_version": "3.30.0",
        "urls": [
            "https://docs.kalshi.com/api-reference/market/get-market-orderbook.md",
            "https://docs.kalshi.com/getting_started/orderbook_responses.md",
            "https://docs.kalshi.com/getting_started/fixed_point_migration.md",
        ],
    },
    "polymarket-markets": {
        "venue": "Polymarket",
        "product": "Polymarket Predictions Gamma metadata, not the US or Perps product",
        "documented_api_version": "1.0.0",
        "urls": [
            "https://docs.polymarket.com/api-reference/markets/list-markets.md",
            "https://docs.polymarket.com/market-data/discover-markets.md",
        ],
    },
    "polymarket-book": {
        "venue": "Polymarket",
        "product": "Polymarket Predictions CLOB token book, native collateral unspecified here",
        "documented_api_version": "1.0.0",
        "urls": [
            "https://docs.polymarket.com/api-reference/market-data/get-order-book.md",
            "https://docs.polymarket.com/market-data/prices-order-books.md",
            "https://docs.polymarket.com/api-reference/rate-limits.md",
        ],
    },
}


class SnapshotError(ValueError):
    """Failed acquisition. Completed pages/attempts remain inspectable in memory."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.completed_pages: tuple[dict[str, Any], ...] = ()
        self.attempts: tuple[dict[str, Any], ...] = ()


class DisallowedRequest(SnapshotError):
    pass


class SchemaError(SnapshotError):
    pass


def _integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SnapshotError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _identifier(value: str, pattern: re.Pattern, name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise SnapshotError(f"invalid {name}")
    return value


def _cursor(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 2048 or not value.isascii() or not value.isprintable():
        raise SchemaError("cursor must be a bounded printable string or null")
    return value


def validate_public_url(url: str) -> str:
    """Allow only four documented GET routes and known, bounded query fields."""
    if not isinstance(url, str) or len(url) > 4096 or not url.isascii() or any(ord(c) < 33 for c in url):
        raise DisallowedRequest("invalid public URL")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise DisallowedRequest("malformed URL") from exc
    if (parsed.scheme != "https" or parsed.fragment or parsed.username is not None
            or parsed.password is not None or "\\" in url or "%" in parsed.path):
        raise DisallowedRequest("HTTPS without userinfo, fragments or encoded paths is required")
    try:
        pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise DisallowedRequest("invalid query string") from exc
    query = dict(pairs)
    if len(query) != len(pairs):
        raise DisallowedRequest("duplicate query keys are forbidden")
    if parsed.netloc == "external-api.kalshi.com" and parsed.path == "/trade-api/v2/markets":
        kind, keys = "kalshi-markets", {"limit", "cursor", "status", "series_ticker", "event_ticker"}
    elif parsed.netloc == "external-api.kalshi.com" and re.fullmatch(
            r"/trade-api/v2/markets/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/orderbook", parsed.path):
        kind, keys = "kalshi-book", {"depth"}
    elif parsed.netloc == "gamma-api.polymarket.com" and parsed.path == "/markets":
        kind, keys = "polymarket-markets", {"limit", "offset", "closed", "order", "ascending"}
    elif parsed.netloc == "clob.polymarket.com" and parsed.path == "/book":
        kind, keys = "polymarket-book", {"token_id"}
    else:
        raise DisallowedRequest("host/path is outside the public snapshot allowlist")
    if set(query) - keys:
        raise DisallowedRequest("unsupported query fields; credentials and arbitrary URLs are forbidden")
    for name, lower, upper in (("limit", 1, 100), ("offset", 0, 1000), ("depth", 1, 100)):
        if name in query:
            if not re.fullmatch(r"[0-9]+", query[name]):
                raise DisallowedRequest(f"invalid {name}")
            _integer(int(query[name]), name, lower, upper)
    for name in ("series_ticker", "event_ticker"):
        if name in query:
            _identifier(query[name], TICKER, name)
    if "cursor" in query:
        _cursor(query["cursor"])
    if "status" in query and query["status"] not in {"open", "unopened", "closed", "settled"}:
        raise DisallowedRequest("unsupported market status")
    if "closed" in query and query["closed"] not in {"true", "false"}:
        raise DisallowedRequest("closed must be true or false")
    if "order" in query and query["order"] != "id":
        raise DisallowedRequest("discovery ordering is fixed to id")
    if "ascending" in query and query["ascending"] != "true":
        raise DisallowedRequest("discovery ordering must be ascending")
    required = {"kalshi-markets": {"limit"}, "kalshi-book": {"depth"},
                "polymarket-markets": {"limit", "offset", "closed", "order", "ascending"},
                "polymarket-book": {"token_id"}}[kind]
    if not required <= set(query):
        raise DisallowedRequest("missing bounded public query parameters")
    if kind == "polymarket-book":
        _identifier(query["token_id"], TOKEN_ID, "token ID")
    return kind


@dataclass(frozen=True)
class RequestPolicy:
    timeout_seconds: float = 10.0
    max_response_bytes: int = 2 * 1024 * 1024
    retries: int = 1
    minimum_interval_seconds: float = 1.0
    max_retry_after_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name, lower, upper in (("timeout_seconds", 0.1, 30), ("minimum_interval_seconds", 1, 60),
                                   ("max_retry_after_seconds", 1, 60)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lower <= value <= upper:
                raise SnapshotError(f"{name} must be finite and in [{lower}, {upper}]")
        _integer(self.max_response_bytes, "max_response_bytes", 1024, 4 * 1024 * 1024)
        _integer(self.retries, "retries", 0, 2)


@dataclass(frozen=True)
class PublicResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    received_at: datetime
    final_url: str


class GetTransport(Protocol):
    def __call__(self, url: str, *, timeout: float, max_bytes: int) -> PublicResponse: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        response.close()
        raise DisallowedRequest(f"HTTP {code} redirect refused; no redirected request was sent")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PublicGetTransport:
    """Direct verified-TLS GET only; no cookies, proxy credentials or redirects.

    Socket timeout plus between-chunk deadline, not an OS-level DNS deadline.
    Environments requiring proxy configuration are intentionally unsupported.
    """

    def __init__(self, *, acknowledge_terms: bool = False) -> None:
        if acknowledge_terms is not True:
            raise DisallowedRequest("explicit terms/permissions acknowledgement is required before public GET transport")

    def __call__(self, url: str, *, timeout: float, max_bytes: int) -> PublicResponse:
        validate_public_url(url)
        RequestPolicy(timeout_seconds=timeout, max_response_bytes=max_bytes, retries=0)
        request = Request(url, method="GET", headers={
            "Accept": "application/json", "Accept-Encoding": "identity",
            "User-Agent": "prediction-market-lab/0.1 public-read-only",
        })
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        deadline = time.monotonic() + timeout
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as exc:
            try:
                return PublicResponse(exc.code, dict(exc.headers), b"", _utc_now(), url)
            finally:
                exc.close()
        with response:
            if response.geturl() != url:
                raise DisallowedRequest("response URL differs from the allowed requested URL")
            headers = {key.lower(): value for key, value in response.headers.items()}
            if headers.get("content-encoding", "identity").lower() != "identity":
                raise SnapshotError("compressed response refused; identity encoding was requested")
            length = headers.get("content-length")
            if length is not None:
                if not length.isascii() or not length.isdecimal() or len(length) > 12 or int(length) > max_bytes:
                    raise SnapshotError("invalid or excessive response Content-Length")
            chunks, total = [], 0
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("response deadline exceeded between chunks")
                chunk = response.read1(min(65536, max_bytes + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise SnapshotError("response exceeds configured byte limit")
                chunks.append(chunk)
            if length is not None and total != int(length):
                raise SnapshotError("response ended before its advertised Content-Length")
            return PublicResponse(response.status, headers, b"".join(chunks), _utc_now(), response.geturl())


def _strict_json(body: bytes) -> Any:
    def no_constant(value):
        raise SchemaError(f"nonfinite JSON constant {value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SchemaError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        payload = json.loads(body.decode("utf-8"), parse_constant=no_constant, object_pairs_hook=unique_pairs)
        json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
        return payload
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise SchemaError(f"invalid or ambiguous UTF-8 JSON: {exc}") from exc


def _string(value: Any, name: str, *, nonempty: bool = True) -> None:
    if not isinstance(value, str) or (nonempty and not value):
        raise SchemaError(f"{name} must be a {'nonempty ' if nonempty else ''}string")


def _native_decimal(value: Any, name: str, *, unit_interval: bool = False) -> None:
    if not isinstance(value, str) or len(value) > 128 or DECIMAL_STRING.fullmatch(value) is None:
        raise SchemaError(f"{name} must remain a nonnegative native decimal string")
    number = Decimal(value)
    if not number.is_finite() or (unit_interval and number > 1):
        raise SchemaError(f"invalid {name}")


def validate_payload(kind: str, payload: Any, url: str) -> dict[str, Any]:
    """Validate the consumed shape while retaining every native field and array order."""
    if validate_public_url(url) != kind:
        raise SchemaError("schema kind does not match the allowlisted request")
    query = dict(parse_qsl(urlsplit(url).query))
    if kind in {"kalshi-markets", "polymarket-markets"}:
        if kind == "kalshi-markets":
            if not isinstance(payload, dict) or "cursor" not in payload or not isinstance(payload.get("markets"), list):
                raise SchemaError("expected markets array and cursor envelope")
            items, next_cursor = payload["markets"], _cursor(payload["cursor"])
            identifier = "ticker"
        else:
            if not isinstance(payload, list):
                raise SchemaError("Gamma /markets must return an array, not an SDK/keyset envelope")
            items, next_cursor, identifier = payload, None, "id"
        if len(items) > int(query["limit"]):
            raise SchemaError("response exceeds requested record limit")
        for item in items:
            if not isinstance(item, dict):
                raise SchemaError("market record must be an object")
            _string(item.get(identifier), identifier)
            if kind == "kalshi-markets":
                for field in ("event_ticker", "market_type", "status", "rules_primary", "rules_secondary"):
                    _string(item.get(field), field, nonempty=not field.startswith("rules_"))
                for field, value in item.items():
                    if field.endswith("_dollars") or field.endswith("_fp"):
                        if value is not None:
                            _native_decimal(value, field)
            else:
                for field in ("question", "conditionId", "slug", "description", "clobTokenIds",
                              "outcomes", "outcomePrices", "denominationToken"):
                    if field in item and item[field] is not None:
                        _string(item[field], field, nonempty=False)
                for field in ("clobTokenIds", "outcomes", "outcomePrices"):
                    if item.get(field):
                        encoded = _strict_json(item[field].encode("utf-8"))
                        if not isinstance(encoded, list) or any(not isinstance(value, str) for value in encoded):
                            raise SchemaError(f"{field} must be a JSON-encoded string array")
                for field in ("closed", "active", "enableOrderBook", "acceptingOrders"):
                    if field in item and item[field] is not None and type(item[field]) is not bool:
                        raise SchemaError(f"{field} must be boolean or null")
        return {"record_count": len(items), "identifiers": [item[identifier] for item in items],
                "next_cursor": next_cursor}
    if not isinstance(payload, dict):
        raise SchemaError("book response must be an object")
    if kind == "kalshi-book":
        native = payload.get("orderbook_fp")
        if not isinstance(native, dict):
            raise SchemaError("expected current orderbook_fp schema, not legacy integer-cent orderbook")
        counts = {}
        for side in ("yes_dollars", "no_dollars"):
            levels = native.get(side)
            if not isinstance(levels, list):
                raise SchemaError(f"{side} must be an array")
            for level in levels:
                if not isinstance(level, list) or len(level) != 2:
                    raise SchemaError("Kalshi levels must be [price_dollars, count_fp]")
                _native_decimal(level[0], "price_dollars", unit_interval=True)
                _native_decimal(level[1], "count_fp")
            if len(levels) > int(query["depth"]):
                raise SchemaError("book response exceeds requested depth")
            counts[side] = len(levels)
        return {"record_count": 1, "level_counts": counts,
                "interpretation": "both native arrays are bids; array ordering is preserved, not assumed"}
    for field in ("market", "asset_id", "timestamp", "hash"):
        _string(payload.get(field), field)
    if payload["asset_id"] != query["token_id"]:
        raise SchemaError("returned asset_id does not match requested token_id")
    if not payload["timestamp"].isdecimal():
        raise SchemaError("CLOB timestamp must retain its native digit string")
    for field in ("min_order_size", "tick_size", "last_trade_price"):
        _native_decimal(payload.get(field), field, unit_interval=field != "min_order_size")
    if type(payload.get("neg_risk")) is not bool:
        raise SchemaError("neg_risk must be boolean")
    counts = {}
    for side in ("bids", "asks"):
        levels = payload.get(side)
        if not isinstance(levels, list):
            raise SchemaError(f"{side} must be an array")
        for level in levels:
            if not isinstance(level, dict):
                raise SchemaError("CLOB levels must be price/size objects")
            _native_decimal(level.get("price"), "price", unit_interval=True)
            _native_decimal(level.get("size"), "size")
        counts[side] = len(levels)
    return {"record_count": 1, "level_counts": counts, "venue_timestamp_native": payload["timestamp"],
            "interpretation": "native bids/asks and ordering retained; collateral and fees are not inferred"}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


class SnapshotClient:
    """Small synchronous public reads, with per-instance request pacing.

    True acknowledgement is necessary, never sufficient proof of eligibility.
    There is deliberately no arbitrary-URL, method, header, credential or proxy option.
    """

    def __init__(self, *, acknowledge_terms: bool = False, policy: RequestPolicy | None = None,
                 transport: GetTransport | None = None, clock: Callable[[], datetime] = _utc_now,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if acknowledge_terms is not True:
            raise DisallowedRequest("explicit acknowledge_terms=True is required; check applicable terms/permissions first")
        self.policy = policy or RequestPolicy()
        self._transport = transport or PublicGetTransport(acknowledge_terms=True)
        self._clock, self._monotonic, self._sleep = clock, monotonic, sleep
        self._next_request_at = 0.0

    def _now(self) -> str:
        return utc(self._clock()).isoformat()

    def _retry_delay(self, headers: Mapping[str, str], attempt: int) -> float:
        delay = max(self.policy.minimum_interval_seconds, float(2 ** attempt))
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                if retry_after.isdecimal():
                    requested = float(retry_after)
                else:
                    date = parsedate_to_datetime(retry_after)
                    if date.tzinfo is None:
                        raise ValueError("timezone missing")
                    requested = max(0.0, (date - utc(self._clock())).total_seconds())
            except (TypeError, ValueError, OverflowError) as exc:
                raise SnapshotError("invalid Retry-After; refusing an early speculative retry") from exc
            delay = max(delay, requested)
        if delay > self.policy.max_retry_after_seconds:
            raise SnapshotError("Retry-After exceeds local wait budget; stop rather than retry early")
        return delay

    def _page(self, kind: str, url: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        if validate_public_url(url) != kind:
            raise DisallowedRequest("endpoint kind mismatch")
        for index in range(self.policy.retries + 1):
            self._sleep(max(0.0, self._next_request_at - self._monotonic()))
            self._next_request_at = self._monotonic() + self.policy.minimum_interval_seconds
            attempt = {"requested_url": url, "method": "GET", "attempt": index + 1,
                       "started_at": self._now()}
            attempts.append(attempt)
            headers: dict[str, str] = {}
            try:
                response = self._transport(url, timeout=self.policy.timeout_seconds,
                                           max_bytes=self.policy.max_response_bytes)
            except (TimeoutError, URLError, OSError, HTTPException) as exc:
                attempt.update({"finished_at": self._now(), "error": type(exc).__name__})
                if index == self.policy.retries:
                    raise SnapshotError(f"public GET failed after bounded attempts: {type(exc).__name__}") from exc
            except SnapshotError as exc:
                attempt.update({"finished_at": self._now(), "error": type(exc).__name__})
                raise
            else:
                attempt.update({"finished_at": self._now(), "http_status": response.status})
                if response.final_url != url:
                    raise DisallowedRequest("redirected/mismatched response URL refused")
                headers = {key.lower(): value for key, value in response.headers.items() if key.lower() in SAFE_HEADERS}
                if response.status in RETRY_STATUSES:
                    if index == self.policy.retries:
                        raise SnapshotError(f"HTTP {response.status}; bounded retries exhausted")
                elif response.status != 200:
                    raise SnapshotError(f"HTTP {response.status}; no authentication or access-bypass fallback")
                else:
                    if len(response.body) > self.policy.max_response_bytes:
                        raise SnapshotError("response exceeds configured byte limit")
                    media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if media_type != "application/json" and not media_type.endswith("+json"):
                        raise SchemaError("expected an explicit JSON response content type")
                    if headers.get("content-encoding", "identity").lower() != "identity":
                        raise SnapshotError("compressed response refused")
                    payload = _strict_json(response.body)
                    checked = validate_payload(kind, payload, url)
                    return {
                        "requested_url": url, "method": "GET", "http_status": response.status,
                        "received_at_utc": utc(response.received_at).isoformat(),
                        "publication_time": None, "publication_history": "UNKNOWN",
                        "response_headers": headers, "raw_body_encoding": "base64",
                        "raw_body_base64": base64.b64encode(response.body).decode("ascii"),
                        "raw_sha256": _digest(response.body), "raw_byte_count": len(response.body),
                        "payload_sha256": _digest(_canonical(payload)), "payload": payload,
                        "validated_summary": checked,
                    }
            delay = self._retry_delay(headers, index)
            attempt["retry_delay_seconds"] = delay
            self._next_request_at = max(self._next_request_at, self._monotonic() + delay)
        raise AssertionError("unreachable bounded request loop")

    def _collect(self, kind: str, base: str, parameters: dict[str, Any], max_pages: int) -> dict[str, Any]:
        _integer(max_pages, "max_pages", 1, 10)
        started = self._now()
        pages, attempts, identifiers, duplicates = [], [], set(), set()
        next_cursor, next_offset = None, None
        seen_cursors = set()
        listing = kind.endswith("markets")
        exhausted = not listing
        try:
            for _ in range(max_pages):
                url = base + "?" + urlencode(sorted(parameters.items()))
                page = self._page(kind, url, attempts)
                pages.append(page)
                summary = page["validated_summary"]
                for identifier in summary.get("identifiers", []):
                    if identifier in identifiers:
                        duplicates.add(identifier)
                    identifiers.add(identifier)
                if not listing:
                    break
                if kind == "kalshi-markets":
                    next_cursor = summary["next_cursor"]
                    if next_cursor is None:
                        exhausted = True
                        break
                    if next_cursor in seen_cursors:
                        raise SchemaError("pagination cursor repeated; refusing a loop")
                    seen_cursors.add(next_cursor)
                    parameters["cursor"] = next_cursor
                else:
                    if summary["record_count"] < parameters["limit"]:
                        exhausted = True
                        break
                    next_offset = parameters["offset"] + parameters["limit"]
                    parameters["offset"] = next_offset
        except SnapshotError as exc:
            exc.completed_pages, exc.attempts = tuple(pages), tuple(attempts)
            raise
        reference = REFERENCES[kind]
        return {
            "format": "prediction-market-lab/public-snapshot-v1",
            "classification": "PUBLIC_VENUE_SNAPSHOT_NOT_SYNTHETIC",
            "venue": reference["venue"], "product": reference["product"], "endpoint": kind,
            "snapshot_scope": "CURRENT_ONLY", "historical_point_in_time": False,
            "publication_history": "UNKNOWN", "live_edge_claim": False,
            "terms_acknowledged": True, "eligibility_or_redistribution_rights_verified": False,
            "collateral_fees_and_payouts": "VENUE_NATIVE_FIELDS_NOT_COERCED_OR_VERIFIED",
            "schema_reference": {"documentation_checked_on": CHECKED_ON,
                                 "documented_api_version": reference["documented_api_version"],
                                 "documentation_urls": reference["urls"],
                                 "validation_scope": "consumed envelope/identity/native string fields; unknown fields preserved"},
            "started_at_utc": started, "completed_at_utc": self._now(),
            "request_policy": asdict(self.policy), "attempts": attempts, "pages": pages,
            "pagination": {
                "mechanism": "cursor" if kind == "kalshi-markets" else "offset" if listing else "none",
                "pages_received": len(pages), "max_pages": max_pages,
                "records_received": sum(page["validated_summary"]["record_count"] for page in pages),
                "page_size": parameters.get("limit"), "truncated": not exhausted,
                "more_records_possible": not exhausted,
                "completion": "response_exhausted" if exhausted and listing else "single_snapshot" if not listing else "page_budget_exhausted",
                "next_cursor": next_cursor, "next_offset": next_offset if not exhausted else None,
                "duplicate_identifiers_preserved": sorted(duplicates),
                "consistency": "mutable paginated reads are not one atomic snapshot; gaps and duplicates remain possible",
            },
            "requested_book_depth": parameters.get("depth"),
            "complete_book_history": False,
            "book_depth_limit_may_omit_levels": kind == "kalshi-book",
            "limitations": [
                "receipt time is local acquisition time, not source publication or historical availability",
                "native response order retained; documentation examples disagree on order, so no best-price index is assumed",
                "metadata/quotes do not establish queue position, fills, fees, collateral equivalence or live edge",
                "permission acknowledgement is not a legal eligibility or redistribution determination",
            ],
        }

    def kalshi_markets(self, *, page_size: int = 20, max_pages: int = 1, status: str = "open",
                       series_ticker: str | None = None, event_ticker: str | None = None) -> dict[str, Any]:
        _integer(page_size, "page_size", 1, 100)
        query: dict[str, Any] = {"limit": page_size, "status": status}
        for name, value in (("series_ticker", series_ticker), ("event_ticker", event_ticker)):
            if value is not None:
                query[name] = _identifier(value, TICKER, name)
        return self._collect("kalshi-markets", KALSHI + "/markets", query, max_pages)

    def kalshi_book(self, ticker: str, *, depth: int = 20) -> dict[str, Any]:
        _identifier(ticker, TICKER, "ticker")
        _integer(depth, "depth", 1, 100)
        return self._collect("kalshi-book", KALSHI + f"/markets/{ticker}/orderbook", {"depth": depth}, 1)

    def polymarket_markets(self, *, page_size: int = 20, max_pages: int = 1,
                           closed: bool = False) -> dict[str, Any]:
        _integer(page_size, "page_size", 1, 100)
        if type(closed) is not bool:
            raise SnapshotError("closed must be an explicit boolean")
        return self._collect("polymarket-markets", GAMMA + "/markets",
                             {"limit": page_size, "offset": 0, "closed": str(closed).lower(),
                              "order": "id", "ascending": "true"}, max_pages)

    def polymarket_book(self, token_id: str) -> dict[str, Any]:
        _identifier(token_id, TOKEN_ID, "token ID")
        return self._collect("polymarket-book", CLOB + "/book", {"token_id": token_id}, 1)


def snapshot_output_path(path: Path) -> Path:
    """Require a new gitignored-suffix snapshot under the current working tree."""
    target, root = path.resolve(), Path.cwd().resolve()
    if not target.is_relative_to(root) or not target.name.endswith(".snapshot.json"):
        raise SnapshotError("output must be inside the working directory and end in .snapshot.json")
    if target.exists():
        raise SnapshotError("snapshot output already exists; overwriting is refused")
    return target


def write_snapshot(snapshot: Mapping[str, Any], path: Path) -> str:
    target = snapshot_output_path(path)
    encoded = (json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with target.open("xb") as handle:
            created = True
            handle.write(encoded)
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return _digest(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OPTIONAL public-GET current snapshots only. No credentials or trading. Help is offline.",
        epilog="Acknowledgement is not proof of eligibility or redistribution rights. Never bypass access restrictions.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in REFERENCES:
        sub = subcommands.add_parser(name, help=REFERENCES[name]["product"])
        sub.add_argument("--acknowledge-terms", action="store_true", required=True,
                         help="affirm that you checked applicable terms/permissions; not a legal determination")
        sub.add_argument("--output", type=Path, required=True, help="new local path ending in .snapshot.json")
        sub.add_argument("--timeout", type=float, default=10, help="socket timeout seconds (0.1-30)")
        sub.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024, help="response cap (1024-4194304)")
        sub.add_argument("--retries", type=int, default=1, help="additional attempts (0-2), never authentication fallback")
        if name.endswith("markets"):
            sub.add_argument("--page-size", type=int, default=20, help="1-100 records per page")
            sub.add_argument("--max-pages", type=int, default=1, help="1-10 pages, truncation explicitly recorded")
        if name == "kalshi-markets":
            sub.add_argument("--status", choices=("open", "unopened", "closed", "settled"), default="open")
            sub.add_argument("--series-ticker")
            sub.add_argument("--event-ticker")
        elif name == "kalshi-book":
            sub.add_argument("--ticker", required=True)
            sub.add_argument("--depth", type=int, default=20, help="1-100 levels per side, never unbounded depth=0")
        elif name == "polymarket-markets":
            sub.add_argument("--closed", action="store_true", help="request closed rather than not-closed records")
        else:
            sub.add_argument("--token-id", required=True)
    args = parser.parse_args(argv)
    try:
        output = snapshot_output_path(args.output)
        policy = RequestPolicy(timeout_seconds=args.timeout, max_response_bytes=args.max_bytes, retries=args.retries)
        client = SnapshotClient(acknowledge_terms=args.acknowledge_terms, policy=policy)
        if args.command == "kalshi-markets":
            result = client.kalshi_markets(page_size=args.page_size, max_pages=args.max_pages, status=args.status,
                                           series_ticker=args.series_ticker, event_ticker=args.event_ticker)
        elif args.command == "kalshi-book":
            result = client.kalshi_book(args.ticker, depth=args.depth)
        elif args.command == "polymarket-markets":
            result = client.polymarket_markets(page_size=args.page_size, max_pages=args.max_pages, closed=args.closed)
        else:
            result = client.polymarket_book(args.token_id)
        checksum = write_snapshot(result, output)
    except (SnapshotError, OSError) as exc:
        print(json.dumps({"status": "ACQUISITION_FAILED", "error": str(exc),
                          "completed_pages_not_written": len(getattr(exc, "completed_pages", ())),
                          "attempts": getattr(exc, "attempts", ()), "synthetic_fallback": False}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "CURRENT_SNAPSHOT_SAVED", "output": str(output),
                      "file_sha256": checksum, "pagination": result["pagination"],
                      "historical_point_in_time": False, "live_edge_claim": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
