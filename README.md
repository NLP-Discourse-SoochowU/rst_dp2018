## Transition based RST-style Discourse Parser

<b>-- General Information</b>
```
   1. This is an RST-style text-level discourse parser based on "shift-reduce" mechanism.
   2. An end-to-end parser version of "RST Discourse Parsing with Tree-structured Neural Networks".
```

<b>-- Required Packages</b>
```
   torch==0.4.0 
   numpy==1.14.1
   nltk==3.3
   stanfordcorenlp==3.9.1.1
```

<b>-- RST Parsing with Raw Documents</b>
```
   1. Prepare your raw documents in data/raw_txt in the format of *.out
   2. Run the Stanford CoreNLP with the given bash script corpus_rst.sh using the 
      command "./corpus_rst.sh "
   3. Run parser.py to parse these raw documents into objects of rst_tree class.
      - segmentation
      - wrap them into trees, saved in "data/trees_parsed/trees_list.pkl"
   4. Run drawer.py to draw those trees out by NLTK
```

<b>-- Training Your Own RST Parser</b>

      TODO

<b>-- Reference</b>

   Please read the following paper for more technical details
   
   [Longyin Zhang, Cheng Sun, Xin Tan, and Fang Kong, RST Discourse Parsing with 
   Tree-structured Neural Networks.](https://link.springer.com/chapter/10.1007/978-981-13-3083-4_2)

<b>-- Developer</b>
```
  Longyin Zhang
  Natural Language Processing Lab, School of Computer Science and Technology, Soochow University, China
  mail to: zzlynx@outlook.com, lyzhang9@stu.suda.edu.cn

```

<b>-- License</b>
```
   Copyright (c) 2018, Soochow University NLP research group. All rights reserved.
   Redistribution and use in source and binary forms, with or without modification, are permitted provided that
   the following conditions are met:
   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
      following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
```
