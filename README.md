# PharMe Study Results

Results and analysis scripts of the
[PharMe validation study](./STUDY-DESCRIPTION.md).

## Preprocessing

Preprocesses surveys on the fly if not done yet.

This includes anonymization and reading information from external sources, such
as REDCap.

## Manually Add Responses

If responses could not be added via ehive, you can add the responses in a CSV
file that is formatted like the original survey file and named
`tasks/[original-survey].manual.csv`.

You will also need to add the time points manually in
`external/participant_surveys.manual.json`.

## Run Script

All content is created in the `analyses` Jupyter Notebook 📒 in the root folder
of this repository.

Setup and run the notebooks as you prefer, I can recommend using VS Code and
Python environments:

* Setup secrets 🤫 and settings ⚙️ in `.env` file (see `.env.example`)
* Optionally download files for preprocessing (code not tested without them but
  all preprocessed content for the analyses is committed):
  * The original `tasks/` data
  * `external/comprehension_data.json` from study backend (created as part of
    lab data preprocessing)
  * The original survey progress data for baseline, case, and control in
    `external`
* Create a new Python 🐍 environment in VS CodeCode
  * Requires Python extension
  * Tested with Python `3.13.3`
  * Should prompt to install `requirements.txt`
* Execute cells in Jupyter Notebooks 📒 using VS Code
  * Requires Jupyter extension
  * Select the just created environment as kernel
  * You should be prompted if kernel packages are missing
