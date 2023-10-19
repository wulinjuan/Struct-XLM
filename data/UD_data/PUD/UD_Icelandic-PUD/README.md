# Summary

Icelandic-PUD is the Icelandic part of the Parallel Universal Dependencies (PUD) treebanks.

# Introduction

The Icelandic-PUD consists of Icelandic translations of 1.000 sentences from the news domain and from Wikipedia. The morphological and syntactic annotation have been manually validated.
Icelandic-PUD was not created and a part of the CoNLL 2017 shared task like the other PUD treebanks.

# Acknowledgments

Translations were produced by Ölvir Gíslason, a professional translator. The automatic tagging was carried out using ABLTagger, which is based on BiLSTM models, a morphological lexicon and lexical category identification. It is developed by Steinþór Steingrímsson, Örvar Kárason and Hrafn Loftsson and available from https://github.com/steinst/ABLTagger. For lemmatizing the high accuracy lemmatizer Nefnir was run, it is developed by Jón Daði Ingólfsson, Svanhvít Lilja Ingólfsdóttir and Hrafn Loftsson and available at https://github.com/jonfd/nefnir. For preprocessing the syntactic annotation, a delexicalized parser was run using UDPipe, developed by Milan Straka, see https://ufal.mff.cuni.cz/udpipe.

The morphological and syntactic annotation were checked and corrected manually by Hildur Jónsdóttir.

# PUD Treebanks

There are 1.000 sentences in each language, always in the same order. (The sentence
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

* 2022-11-15 v2.11
  * Minor fixes of multi-subject clauses.
* 2020-05-15 v2.6
  * Initial release in Universal Dependencies.

<pre>
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: UD v2.6
License: CC BY-SA 4.0
Includes text: yes
Genre: news wiki
Lemmas: automatic with corrections
UPOS: automatic with corrections
XPOS: automatic with corrections
Features: automatic with corrections
Relations: automatic with corrections
Contributors: Jónsdóttir, Hildur
Contributing: here
Contact: hildur.jonsdottir@gmail.com
===============================================================================
</pre>


