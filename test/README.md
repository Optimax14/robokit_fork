Created by Itay Kadosh at IRVL Lab @ University of Texas at Dallas

This README file is to document changes done to the original repository

```utils.py```
* Added filter and save_mask methods
  *   filter: Uses multiple factors to eliminate false detections and increase accuracy of model.
  *   save_mask: Saves the masks created by SAM model into a format including a main folder called segments, a subfolder with the number of the image
      * subfolders for each class (chair, table, etc.)

```perception.py```
* changed the predict function to include the box_threshold and text_threshold as parameters for ease of use.

