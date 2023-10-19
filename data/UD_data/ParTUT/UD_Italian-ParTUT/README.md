# Summary

UD_Italian-ParTUT is a conversion of a multilingual parallel treebank developed at the University of Turin, 
 and consisting of a variety of text genres, including talks, legal texts and Wikipedia articles, among others.


# Introduction

UD_Italian-ParTUT data is derived from the already-existing parallel treebank Par(allel)TUT.

ParTUT is a morpho-syntactically annotated collection of Italian/French/English parallel sentences, 
which includes texts from different sources and representing different genres and domains, released in several formats.

ParTUT comprises approximately 167,000 tokens, with an average amount
of 2,100 sentences per language. The texts of the collection currently available were
gathered from a large number of sources and domains:
* the [Creative Commons](http://creativecommons.org/licenses/by-nc-sa/2.0) open license;
* the [DGT-Translation Memory](https://ec.europa.eu/jrc/en/language-technologies/dgt-translation-memory)
* the [Europarl](http://www.statmt.org/europarl/) parallel corpus (section ep_00_01_17);
* publicly available pages from [Facebook website](https://www.facebook.com/help/345121355559712/);
* the [JRC-Acquis multilingual parallel corpus](http://optima.jrc.it/Acquis/index_2.2.html) (section jrc52006DC243);
* several articles from [Project Syndicate©](https://www.project-syndicate.org/) [ABSENT IN UD_French-ParTUT];
* the [Universal Declaration of Human Rights](http://www.ohchr.org/EN/UDHR/Pages/SearchByLang.aspx);
* Wikipedia articles retrieved in the English section and then translated into Italian only by graduate students in Translation  Studies [ABSENT IN UD_French-ParTUT];
* the [Web Inventory of Translated Talks](https://wit3.fbk.eu/mt.php?release=2012-02) .

ParTUT data can be downloaded [here](http://www.di.unito.it/~tutreeb/treebanks.html) and [here](https://github.com/msang/partut-repo).


NOTE: While the Italian section of ParTUT is already included in [UD_Italian](https://github.com/UniversalDependencies/UD_Italian/tree/master), UD_Italian-ParTUT comprises just those sentences having a 1:1 correspondence with their English and French counterparts.

# Acknowledgements
We are deeply grateful to Project Syndicate© for letting us download and exploit their articles as text material, under the terms of educational use. 


# Corpus splitting

Since version 2.1, the corpus has been re-partitioned so as to avoid overlapping sentences with UD_Italian. 
The treebank has thus been randomly split as follows:

* it_partut-ud-train.conllu: 48976 words (1783 sentences)
* it_partut-ud-dev.conllu: 4871 words (238 sentences)
* it_partut-ud-test.conllu: 1711 words (69 sentences)

In order to preserve the 1:1 correspondence among the three language sections, all of them were partitioned in the same way; therefore the same sentences, in the same order,
are found in the training, development and test set of UD_French-ParTUT and UD_English-ParTUT as well.


# Basic statistics

* Tree count:  2090
* Word count:  55558
* Token count: 51614
* Dep. relations: 44 of which 11 language specific
* POS tags: 15
* Category=value feature pairs: 37
		

# References
 
* Manuela Sanguinetti, Cristina Bosco. 2014. PartTUT: The Turin University Parallel Treebank. 
  In Basili, Bosco, Delmonte, Moschitti, Simi (editors) Harmonization and development of resources and tools for Italian Natural Language Processing within the PARLI project, LNCS, Springer Verlag
  
* Manuela Sanguinetti, Cristina Bosco. 2014. Converting the parallel treebank ParTUT in Universal Stanford Dependencies. 
  In Proceedings of the 1rst Conference for Italian Computational Linguistics (CLiC-it 2014), Pisa (Italy)
  
* Cristina Bosco, Manuela Sanguinetti. 2014. Towards a Universal Stanford Dependencies parallel treebank. 
  In Proceedings of the 13th Workshop on Treebanks and Linguistic Theories (TLT-13), Tubingen (Germany)
  
# Changelog
2019-11-15 v2.5
* no changes since last version

2019-05-15 v2.4
* various corrections to pass new validation

2018-11-15 v2.3
* no changes since last version

2018-04-15 v2.2
* revision of object relatives
 
2017-11-15 v2.1 		
* dates were revised and annotated as flat structures
* change of xpos for copulas (from VA to V)
* other minor corrections 
* revised splits in order to avoid overlapping sentences with UD_Italian

2017-03-01 v2
* initial release


=== Machine-readable metadata ================================================
```
Data available since: UD v2.0
License: CC BY-NC-SA 4.0
Includes text: yes
Genre: legal news wiki
Lemmas: converted with corrections
UPOS: converted with corrections
XPOS: converted with corrections
Features: converted with corrections
Relations: converted with corrections
Contributors: Bosco, Cristina; Sanguinetti, Manuela
Contributing: elsewhere
Contact: msanguin@di.unito.it
```
===============================================================================
