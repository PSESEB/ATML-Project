# ATML-Project
ATML Project Eurlex Dataset MultiClass Problem


## Sebastian's Folder:
  Download arff File and eurovoc qrel\
  Put in same Folder as 3 Python files
  1. Execute preprocess.py
  2. call preprocessLabels.py
  3. Use BuildVectors.py to build number vectors

## Chetan's Folder:
  
  ### For Old Files (eurlex_html_EN_NOT.zip)
  
  Note: Install Libraries like *pandas, *nltk, *bs4 and *pathlib before running the code! 
  
  Download the dataset zip file from: http://www.ke.tu-darmstadt.de/files/resources/eurlex/eurlex_html_EN_NOT.zip \
  Put in same Folder as the .ipynb files
  1. Run cells one by one in text_scraping (zipped_html).ipynb
  2. You'll get a file named "output.csv" in your folder containing all the scraped text and labels (uncleaned!).
  2. Run cells one by one in text_cleaning (zipped_html).ipynb
  3. You'll get a file named "output.csv" in your folder containing all the cleaned text and labels.

  ### For New Files (Downloaded from eurlex_download_EN_NOT.sh script)
  
  Note: Install Libraries like *pandas, *nltk, *bs4 and *pathlib before running the code! 
  
  Download the dataset zip file from: https://drive.google.com/open?id=1NmCggWwJT3W-SBeAQDcA3FH44IiJzCH8 \
  Put in same Folder as the .py files
  1. Run text_scraping_htmlwebscript.py
  2. You'll get a file named "final_scraped.csv" in your folder containing all the scraped text and labels (uncleaned!).
  2. Run text_cleaning_htmlwebscript.py
  3. You'll get a file named "final_cleaned.csv" in your folder containing all the cleaned text and labels.
  
  *You can find the final cleaned data in a .csv file here: https://drive.google.com/open?id=1u_TtIx7L3NeedbeJdJKBDf5WTpfeHIRi
