# LSTM: Sequence to Sequence - Grapheme to Phoneme
## Author: Sree Teja Simha G
## email: Sreetejasimha.Gemaraju@utdallas.edu

### File Structure
The reports directory has the report in .pdf format and also LATEX files related to it. The Sources directory has all source files needed to run. 

The Final directory inside Sources has final 3 versions of which run2.py is to be taken for consideration. The ph.txt is a preprocessed subset of data as mentioned in the report. It is a tab seperated table of words and constituent phonemes. This dataset is split 50:50 
for both training and testing. 

The .ipynb files inside the Source directory are my work in progress python notebook files. It gives a glimpse into how I developed the solution. It is vaguely documented.

### Execution
Inside Final directory run...
`python run2.py`

It shall show training and testing steps. All the output generated is stored in output_file.txt. The same can be done to run3.py and run.py. They are just different configurations of the same model.