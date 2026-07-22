# `rename_images_from_excel.py`

Dataset-preparation utility that pairs images with names read from an Excel workbook and renames the image files in a controlled order. It supports a dry run, collision checks, filename sanitization, optional recursive folder processing, and a CSV mapping of old to new names.

Run it before dataset splitting or model training when filenames need to match an external annotation list. It is not imported by the DL or SL pipelines.
