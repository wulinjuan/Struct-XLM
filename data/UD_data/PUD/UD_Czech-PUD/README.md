# Summary

This is a part of the Parallel Universal Dependencies (PUD) treebanks created
for the [CoNLL 2017 shared task on Multilingual Parsing from Raw Text to
Universal Dependencies](http://universaldependencies.org/conll17/).


# Introduction

There are
1000 sentences in each language, always in the same order. (The sentence
alignment is 1-1 but occasionally a sentence-level segment actually consists
of two real sentences.) The sentences are taken from the news domain (sentence
id starts in ‘n’) and from Wikipedia (sentence id starts with ‘w’). There are
usually only a few sentences from each document, selected randomly, not
necessarily adjacent. The digits on the second and third position in the
sentence ids encode the original language of the sentence. The first 750
sentences are originally English (01). The remaining 250 sentences are
originally German (02), French (03), Italian (04) or Spanish (05) and they
were translated to other languages via English. Translation into German,
French, Italian, Spanish, Arabic, Hindi, Chinese, Indonesian, Japanese,
Korean, Portuguese, Russian, Thai and Turkish has been provided by DFKI and
performed (except for German) by professional translators. Then the data has
been annotated morphologically and syntactically by Google according to Google
universal annotation guidelines; finally, it has been converted by members of
the UD community to UD v2 guidelines.

Additional languages have been provided (both translation and native UD v2
annotation) by other teams: Czech by Charles University, Finnish by University
of Turku and Swedish by Uppsala University.

The entire treebank is labeled as test set (and was used for testing in the
shared task). If it is used for training in future research, the users should
employ ten-fold cross-validation.


# Changelog

* 2022-05-15 v2.10
  * Added VerbForm=Part|Voice=Pass to long forms of passive participles.
  * Added VerbForm=Vnoun to verbal nouns.
  * The word 'každý' is now DET instead of ADJ.
  * The verb 'být' is now AUX in all contexts.
  * Merged PRON/DET 'sám', 'samý'.
* 2020-05-15 v2.6
  * Genitive, dative and instrumental nominals are now considered oblique.
  * Added enhanced graphs (shared parents in coordination).
  * Added enhanced relations with case information.
  * Added enhanced relations around relative clauses.
  * Added enhanced external subjects in control verb constructions.
  * Added empty nodes to enhanced graphs (but orphans are just converted to dep).
* 2019-05-15 v2.4
  * Manually fixed errors, mostly "více/méně" as DET(quantifiers) instead of ADV
* 2018-11-15 v2.3
  * Added LDeriv for passive participles (the infinitive of the source verb).
* 2017-11-15 v2.1
  * First official release after it was used as a surprise dataset in the
    CoNLL 2017 shared task.


=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.1
License: CC BY-SA 4.0
Includes text: yes
Genre: news wiki
Lemmas: converted from manual
UPOS: converted from manual
XPOS: manual native
Features: converted from manual
Relations: manual native
Contributors: Kettnerová, Václava; Hajič jr., Jan; Cinková, Silvie; Urešová, Zdeňka; Straka, Milan; Hajič, Jan; Hlaváčová, Jaroslava; Zeman, Daniel
Contributing: here
Contact: zeman@ufal.mff.cuni.cz
===============================================================================
