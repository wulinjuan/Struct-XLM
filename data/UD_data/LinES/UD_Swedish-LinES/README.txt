# Summary

UD Swedish_LinES is the Swedish half of the LinES Parallel Treebank with UD annotations.
All segments are translations from English and the sources cover literary genres, online
manuals and Europarl data.

# Introduction

UD Swedish_LinES is the Swedish half of the LinES Parallel Treebank with UD annotations.
All segments are translations of the corresponding English segments found in the UD English_LinES
treebank.The original dependency annotation was first automatically converted
to Universal Dependencies and then partially reviewed (Ahrenberg, 2015). In January-February 2017
it was converted to UD version 2 and again reviewed for errors. With version 2.1 lemmata and
morphological features have been added.

The treebank is being developed continuously.

# Acknowledgements

Three of the source texts were collected as part of the Linköping Translation Corpus Corpus
(Merkel, 1999). The treebank was first developed in the project 'Micro- and macro-level
analysis of translations' funded by the Swedish Research Council (Ahrenberg, 2007).

# Details on the sources

All sub-corpora have English originals with Swedish translations. Six of them
are literary works:

Paul Auster: Stad av glas [City of Glass], Tiden, 1995. Translation by
Ulla Roseen.

Saul Bellow: Jerusalem tur och retur [To Jerusalem and back: a
personal accunt], Bonniers, 1977. Translation by Caj Lundgren.

Joseph Conrad: Mörkrets hjärta [Heart of darkness], Wahlström &
Widstrand, Stockholm, 1983. Translation by Margaretha Odelberg.

Nadine Gordimer: Hedersgästen [A Guest of Honour], Bonniers,
1991. Translation by Magnus K:son Lindberg.

J. K. Rowling: Harry Potter och Hemligheternas kammare [Harry Potter
and the Chamber of Secrets], Tiden, 2001. Translation by Lena
Fries-Gedin.

Jennette Winterson: Vintergatan går genom magen [Gut Symmetries], 
Bakhåll, 2017. Translation by Ulla Roseen.

In addition the corpus includes segments from Microsoft Access 2002
Online Help and the Swedish part of the Europarl corpus (v.7).

DATA SPLITS

For version 2.0  about 20% of the trees were randomly selected as test set, 20% as development set, and the rest as training set. This partitioning has remained the same since then.

The partition applies in the same way to the English trees so that the order of corresponding trees is the same in the English and Swedish LinES files. The files are named

 - sv_lines-ud-dev.conllu
 - sv_lines-ud-train.conllu
 - sv_lines-ud-test.conllu


BASIC STATISTICS

Tree count:  4564
Word count:  79812
Token count: 79812
Dep. relations: 40 of which 7 are language-specific
POS tags: 17
Category=value feature pairs: 0


TOKENIZATION

The tokenization is largely based on whitespace, but punctuation marks
except word-internal hyphens are treated as separate tokens. The
original file also has several multi-word tokens, but these are
separated in the UD version with all parts except the first assigned
the UD dependency function 'fixed'. No tokens have internal blanks.


MORPHOLOGY

The morphological annotation in the UFEATS column is copied from the UD_Swedish
treebank where overlaps occur. For other tokens it has been converted from the
morphological information in the original treebank (found in the XPOS column).
Nouns are annotated for case, number, species and gender. Verbs are annotated for
mood, verb form, tense and diathesis, adjectives for case, degree, definiteness, and
number. Pronouns are sub-divided in the morphological description into
Personal, Demonstrative, Interrogative, Indefinite, Relative, Total,
and Expletive, and are annotated for Case and Number, when relevant.

The mapping from language-specific part-of-speech tags to universal tags
was done automatically. There are no other tags than universal tags, but
there may be errors.

SYNTAX

The syntactic annotation in the Swedish UD treebank follows the general
guidelines but adds some language-specific relations:

- nmod:poss
- acl:relcl
- acl:cleft
- compound:prt
- nsubj:pass
- aux:pass
- obl:agent
- csubj:pass

The syntactic annotation was first automatically converted from the original
LinES annotation scheme as described in Ahrenberg (2015). After conversion to UD version 2.0
the analyses have been reviewed again. Occasional deviations from the guidelines may remain.


REFERENCES

Lars Ahrenberg, 2007. LinES: An English-Swedish Parallel
Treebank. Proceedings of the 16th Nordic Conference of Computational
Linguistics (NODALIDA, 2007).

Lars Ahrenberg, 2015. Converting an English-Swedish Parallel Treebank
to Universal Dependencies. Proceedings of the Third International
Conference on Dependency Linguistics (DepLing 2015), Uppsala, August
24-26, 2015, pp. 10-19. ACL Anthology W15-2103.

Magnus Merkel, 1999: Understanding and enhancing translation by
parallel text processing. Linköping Studies in Science and Technology,
Dissertation No. 607.


Changelog

  From version 1.3 to version 2.0 the following changes have been made:
  - a new split of the treebank into train, dev and test data
  - addition of sentence id:s and text comment for every tree
  - addition of document boundaries
  - addition of SpaceAfter=No features in the MISC column
  - more fixed phrases have been recognized as such
  - sentences of the form '"I'm hungry", said John', where the root previously was 'said' have been reanalyzed
    in accordance with the UD guidelines, 'said' then linking to 'hungry' as 'parataxis'

  From version 2.0 to version 2.1 the following changes have been made:
  - all tokens have received a lemma and morphological features have been added to tokens that should have them.
  - the test data have been manually reviewed to fix errors and agree better with the version 2 guidelines.
    The changes affect some 9% of all tokens and 28% of all punctuation tokens.

  Changes for version 2.2, made in order to harmonize annotations with those of UD_Swedish_Talbanken
  - the relative pronoun 'som' has been recategorized as PRON and its dependencies have been changed accordingly to nsubj, nsubj:pass, obj, obl, dislocated as is contextually appropriate
  - adpositions that introduce a clause have had their dependency changed from 'case' to 'mark'
  - cleft sentences of the form 'EXPL är/var XP som ...' have been reanalyzed so that the head word of XP is annotaded as 'root' while the clause introduced by 'som' is annotated as 'acl:cleft'
  In addition many inconsistencies and errors have been rectified.
  
  From version 2.2 to version 2.3
  English names, esp. of software products, such as Microsoft Office 2002, have been reanalysed so as to agree with the analysis
  in the English_Lines treebank. Found errors in the first parts of all three files have been corrected.

  From version 2.3 to version 2.4 all changes concern the correction of errors so as to meet the stricter conditions on      validation.
  
  From version 2.4 to version 2.5. Extension of 679 sentences from Winterson's 'Vintergatan går genom magen'. They have been distributed with 120 sentences for development and test, respectively, and the rest to the training part. The lemmatisation has been further harmonised with Swedish_Talbanken.
  
  From version 2.5 to version 2.6 only minor error corrections, in particular regarding the features PronType, Gender and Number.
  
  For version 2.9 the negative adverb 'inte' has consistently been given the UPOS PART. Also a few error corrections.

--- Machine readable metadata ---

Data available since: UD v1.3
License: CC BY-NC-SA 4.0
Includes text: yes
Genre: fiction nonfiction spoken
Lemmas: converted from manual
UPOS: converted with corrections
XPOS: manual native
Features: automatic
Relations: converted with corrections
Contributors: Ahrenberg, Lars
Contributing: elsewhere
Contact: lars.ahrenberg@liu.se
