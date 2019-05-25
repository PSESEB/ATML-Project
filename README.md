# ATML-Project
ATML Project Eurlex Dataset MultiClass Problem


## Sebastian Folder:
  Download arff File and eurovoc qrel\
  Put in same Folder as 3 Python files
  1. Execute preprocess.py
  2. call preprocessLabels.py
  3. Use BuildVectors.py to build number vectors

## Chetan's Folder:
  
  **Install Libraries like Pandas, nltk, bs4 and pathlib before running the code! 
  
  Download the zip file from: http://www.ke.tu-darmstadt.de/files/resources/eurlex/eurlex_html_EN_NOT.zip
  Put in same Folder as the .ipynb files
  1. Run cells one by one in text_scraping (zipped_html).ipynb
  2. You'll get a file named "output.csv" in your folder containing all the scraped text and labels (uncleaned!).
  2. Run cells one by one in text_cleaning (zipped_html).ipynb
  3. You'll get a file named "output.csv" in your folder containing all the cleaned text and labels.
