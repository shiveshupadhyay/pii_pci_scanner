"""Sensitive-data scanner: file type detection -> extraction (structured / documents / OCR) -> detection -> merge."""

# Stamped on every finding. Bump it whenever a change can alter the output, because results cached downstream
# (see DESIGN.md, incremental scans) are keyed on it.
DETECTOR_VERSION = "1.0.0"
