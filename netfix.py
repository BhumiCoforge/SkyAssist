"""Network fix: make Python skip IPv6 for chosen hosts (see prefer_ipv4)."""
import socket

# Remember the real function even if this module is loaded twice (the patched one carries a marker).
_real_getaddrinfo = getattr(socket.getaddrinfo, "_original", socket.getaddrinfo)
_ipv4_only_hosts: set[str] = set()


def _getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host in _ipv4_only_hosts:
        family = socket.AF_INET
    return _real_getaddrinfo(host, port, family, type, proto, flags)


_getaddrinfo._original = _real_getaddrinfo


def prefer_ipv4(host: str | None) -> None:
    """On some networks the gateway name also resolves to an IPv6 address that never answers.
    Python then waits for the whole connect timeout (30-60 s) before it tries IPv4, so every AI call
    is slow. Asking for IPv4 only, for this one host, avoids the wait. Other hosts are not touched."""
    if not host:
        return
    _ipv4_only_hosts.add(host)
    if socket.getaddrinfo is not _getaddrinfo:
        socket.getaddrinfo = _getaddrinfo