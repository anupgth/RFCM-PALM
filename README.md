# RFCM-PALM: _In-silico_ prediction of S-palmitoylation sites in the synaptic proteins

RFCM-PALM  can predict the palmitoylated cysteine sites on synaptic proteins of mouse sequence data. We present a random forest (RF) classifier-based consensus strategy, for palmitoylated cysteine site prediction with efficient feature selection strategies (KB: K-Best, GA: Genetic Algorithm, and UN: Union of KB and GA) on synaptic proteins of three categories of sex independent mouse datasets (Male, Female, and Combined).

# Usage
####  RFCM-PALM is developed in Python >= 3.
######        python RFCM_PALM.py \<filename\> \<dataType\>
######        Example     : python RFCM_PALM.py inputSeq.fasta Male
######        \<filename\>  : input fasta file     : (example: inputSeq.fasta)
######        \<dataType\>  : Male/Female/Combined : (example: Male)
