"""Streamlit-facing UI helpers shared by the web app."""
from pathlib import Path

# NASA's Expedition 59 crew portrait (public domain; tests/fixtures/SOURCE.md), offered to visitors
# who have no photo of their own to hand.
SAMPLE_PHOTO = Path(__file__).with_name("sample_photo.jpg")
SAMPLE_PHOTO_NAME = "nasa_expedition_59_crew.jpg"
SAMPLE_PHOTO_CREDIT = "Sample photo: NASA, Expedition 59 crew portrait (iss059-s-002), public domain."
