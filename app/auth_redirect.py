from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def frontend_redirect_url(base_url: str, access_token: str, refresh_token: str) -> str:
    """Google 로그인 후 프론트 콜백 페이지로 보낼 URL (프론트는 쿼리의 access_token/refresh_token 을 읽는다)."""
    parts = urlparse(base_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query += [("access_token", access_token), ("refresh_token", refresh_token)]
    return urlunparse(parts._replace(query=urlencode(query)))
