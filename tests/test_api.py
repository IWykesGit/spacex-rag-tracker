# import pytest 
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "SpaceX RAG Tracker" in response.text
    assert "Ask" in response.text  # checks for the form button

def test_launches_endpoint():
    response = client.get("/launches")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "date_utc" in data
    assert "success" in data

@patch("main.get_index")  # mock the @property index
def test_ask_endpoint(mock_get_index):
    # Create a fake response object
    mock_response = MagicMock()
    mock_response.response = "Mocked answer: Boostback burn lasted 48 seconds."
    mock_response.source_nodes = []
    # Add a fake source chunk
    fake_node = MagicMock()
    fake_node.node.get_text.return_value = "Source from ift5_transcript.txt"
    mock_response.source_nodes.append(fake_node)

    # Fake query_engine that returns our mock_response when .query() is called
    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = mock_response

    # Fake index object that returns our fake query_engine
    mock_index_obj = MagicMock()
    mock_index_obj.as_query_engine.return_value = mock_query_engine

    # When get_index() is called, return our fake index object
    mock_get_index.return_value = mock_index_obj

    # Now make the request
    response = client.get("/ask?question=How%20long%20was%20the%20boostback%20burn?")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "48 seconds" in data["answer"]
    assert "sources" in data
    assert len(data["sources"]) > 0