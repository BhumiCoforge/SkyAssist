import socket

import netfix


def test_prefer_ipv4_only_changes_the_chosen_host():
    netfix.prefer_ipv4("localhost")
    infos = socket.getaddrinfo("localhost", 80, 0, socket.SOCK_STREAM)
    assert infos and all(i[0] == socket.AF_INET for i in infos)
    # a host that was not chosen is resolved as before (an IP literal still works)
    assert socket.getaddrinfo("127.0.0.1", 80, 0, socket.SOCK_STREAM)


def test_prefer_ipv4_ignores_empty_host():
    netfix.prefer_ipv4(None)
    netfix.prefer_ipv4("")