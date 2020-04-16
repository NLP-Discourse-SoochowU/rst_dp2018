
==============================================================================================================
                                         RST-style Discourse Parser
==============================================================================================================


-- LICENSE
   Copyright (c) 2018, Longyin Zhang, Soochow University
   All rights reserved.

   Redistribution and use in source and binary forms, with or without modification, are permitted provided that
   the following conditions are met:
   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
      following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.


-- GENERAL INFORMATION
   1. This RST-style discourse parser produces discourse tree structure on full-text level, given a raw text.
   2. The method we use in this parser is similar to the method in the paper RST Discourse Parsing with Tree-structured Neural Networks.


-- REQUIRED PACKAGES
   torch==0.4.0, numpy==1.14.1, nltk==3.3, stanfordcorenlp==3.9.1.1


-- RST PARSING with RAW DOCUMENTS
   1. Prepare your raw documents in data/raw_txt in the format of *.out
   2. Run the Stanford CoreNLP with the given bash script corpus_rst.sh with the command "./corpus_rst.sh "
   3. Run parser.py to parse these raw documents into objects of the class rst_tree (Wrap them into trees) and draw them out.
      - segmentation
      - wrap them into trees, saved in "data/trees_parsed/trees_list.pkl"
   4. Run drawer.py to draw those trees out by NLTK


-- Training Your Own RST Parser
    TODO


-- REFERENCE
   Please read the following paper for more technical details
   RST Discourse Parsing with Tree-structured Neural Networks.


-- DEVELOPERS
  Longyin Zhang
  Natural Language Processing Lab, School of Computer Science and Technology, Soochow University, China
  mail to: zzlynx@outlook.com, lyzhang9@stu.suda.edu.cn