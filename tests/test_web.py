"""End-to-end tests for the web UI server.

Starts the server on an ephemeral port in a background thread and exercises
the public API + static file routes.
"""

import json
import threading
import urllib.request
import urllib.parse

import pytest

from gerrymander.web import serve


@pytest.fixture(scope="module")
def server():
    httpd = serve(host="127.0.0.1", port=0)  # 0 -> OS picks a port
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read()


def _get_json(url):
    status, body = _get(url)
    assert status == 200
    return json.loads(body.decode("utf-8"))


def test_index_serves(server):
    status, body = _get(server + "/")
    assert status == 200
    assert b"<title>Gerrymander" in body


def test_static_leaflet(server):
    status, body = _get(server + "/static/leaflet.css")
    assert status == 200
    assert b".leaflet" in body


def test_static_app_js(server):
    status, body = _get(server + "/static/app.js")
    assert status == 200
    assert b"loadStates" in body


def test_states_api(server):
    data = _get_json(server + "/api/states")
    assert isinstance(data, list)
    assert len(data) == 50
    pa = next(s for s in data if s["code"] == "PA")
    assert pa["name"] == "Pennsylvania"
    assert pa["default_seats"] == 17


def test_plan_api_neutral(server):
    data = _get_json(server + "/api/plan?state=PA&party=neutral&seats=17&seed=7")
    assert data["state"] == "PA"
    assert data["party"] == "neutral"
    assert data["seats"] == 17
    assert "geojson" in data
    assert data["geojson"]["type"] == "FeatureCollection"
    assert len(data["geojson"]["features"]) > 50  # PA has 67 counties
    f = data["geojson"]["features"][0]
    assert "district_id" in f["properties"]
    assert "county" in f["properties"]
    m = data["metrics"]
    for k in ("efficiency_gap", "mean_median_D", "partisan_bias_D",
              "D_seats", "R_seats", "n_districts", "D_vote_share"):
        assert k in m


def test_plan_api_gerrymander(server):
    data_n = _get_json(server + "/api/plan?state=NC&party=neutral&seats=14&seed=3")
    data_r = _get_json(server + "/api/plan?state=NC&party=R&seats=14&intensity=0.9&seed=3")
    # The R-gerrymandered plan should not give R fewer seats than the neutral one.
    assert data_r["metrics"]["R_seats"] >= data_n["metrics"]["R_seats"]
    assert data_r["party"] == "R"


def test_plan_api_seat_count_changes(server):
    base = _get_json(server + "/api/plan?state=OH&party=neutral&seats=15&seed=1")
    small = _get_json(server + "/api/plan?state=OH&party=neutral&seats=8&seed=1")
    assert int(base["metrics"]["n_districts"]) == 15
    assert int(small["metrics"]["n_districts"]) == 8


def test_plan_api_bad_state(server):
    # Server should respond 500 with an error message, not crash.
    try:
        _get(server + "/api/plan?state=ZZ&party=neutral&seats=2")
        assert False, "expected error"
    except urllib.error.HTTPError as e:
        assert e.code in (400, 500)
