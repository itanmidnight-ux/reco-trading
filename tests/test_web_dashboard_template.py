from web_site.dashboard_server import create_app


def test_dashboard_index_renders_without_template_errors() -> None:
    app = create_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Reco Trading" in response.data
