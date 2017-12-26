# ML-StockAnalysisProject
Analysis of Stock Patterns using Machine Learning Techniques in a (hopefully) modular way


This project was made through Eclipse, and therefore has all the implications that follow that (as I am currently unaware of the full implications of this, I won't be going into any detail as to what these are). As this is an in-progress project, this README is not designed as a finished, user-friendly product. Therefore, if this is seen by an end user, please raise an issue regarding it. Thank you. 

Non-Standard Python3.5 package list:
* mysql-connector-python

# Other usage instructions:
* To fit a local version of MySQL, the file config.ini must be updated. It is located in configuration_files
  * Along with this, for now, that MySQL database must contain a table called stock_list, with the columns ticker, yahoo, and google. Other columns are optional, but will not impact the usage of this program. Soon, this table will be created automatically. 
