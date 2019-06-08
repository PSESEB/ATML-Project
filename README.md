# ATML-Project
ATML Project Eurlex Dataset MultiClass Problem


# Sebastian's Folder:
  Download arff File and eurovoc qrel\
  Put in same Folder as 3 Python files
  1. Execute preprocess.py
  2. call preprocessLabels.py
  3. Use BuildVectors.py to build number vectors

# Chetan's Folder:
  
  ### For Old Files (eurlex_html_EN_NOT.zip)
  
  Note: Install Libraries like *pandas, nltk, bs4 and pathlib* before running the code! 
  
  Download the dataset zip file from: http://www.ke.tu-darmstadt.de/files/resources/eurlex/eurlex_html_EN_NOT.zip \
  Put in same Folder as the .ipynb files
  1. Run cells one by one in text_scraping (zipped_html).ipynb
  2. You'll get a file named "output.csv" in your folder containing all the scraped text and labels (uncleaned!).
  2. Run cells one by one in text_cleaning (zipped_html).ipynb
  3. You'll get a file named "output.csv" in your folder containing all the cleaned text and labels.

  ### For New Files (Downloaded from eurlex_download_EN_NOT.sh script)
  
  Note: Install Libraries like *pandas, nltk, bs4 and pathlib* before running the code! 
  
  Download the dataset zip file from: https://drive.google.com/open?id=1NmCggWwJT3W-SBeAQDcA3FH44IiJzCH8 \
  Put in same Folder as the .py files
  1. Run text_scraping_htmlwebscript.py
  2. You'll get a file named "final_scraped.csv" in your folder containing all the scraped text and labels (uncleaned!).
  3. Run text_cleaning_htmlwebscript.py
  4. You'll get a file named "final_cleaned.csv" in your folder containing all the cleaned text and labels.
  5. Run remove_labels.py or duplicate_labels.py on "final_cleaned.csv" to sort imbalance issues.
  6. You'll get a file named "imabalanced_labeles_removed.csv" or "imbalanced_labels_duplicated.csv" based on your script selection from step 5. 
  
  *You can find the final cleaned data in a .csv file here: https://drive.google.com/open?id=1u_TtIx7L3NeedbeJdJKBDf5WTpfeHIRi* \  
  *You can find the sorted data in a .csv file here: https://drive.google.com/open?id=1cJQiNfzbkKRwRs8TdrQNQBDPXecb9M4Y*
