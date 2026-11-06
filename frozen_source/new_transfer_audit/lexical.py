"""Same frozen lexical implementation as the preceding audit; CPU only."""
import collections,math,re
import numpy as np
from scipy.sparse import csr_matrix
from common import feature_string,compact,digest

class LexicalIndex:
 """Raw-count TF times smoothed IDF, Unicode words, unigram/bigram, L2."""
 def __init__(self,bank):
  self.bank=bank;counts=[self.terms(feature_string(r)) for r in bank];df=collections.Counter()
  for c in counts:df.update(c.keys())
  self.vocabulary=sorted(df);self.position={w:i for i,w in enumerate(self.vocabulary)}
  self.idf=np.array([math.log((1+len(bank))/(1+df[w]))+1 for w in self.vocabulary])
  self.matrix=self.encode_counts(counts)
  self.audit={'documents':len(bank),'vocabulary_terms':len(self.vocabulary),
   'word_regex':r'(?u)\b\w+\b','lowercase':True,'ngram_range':[1,2],'tf':'raw count',
   'idf':'log((1+number_of_bank_records)/(1+document_frequency))+1','norm':'l2',
   'vocabulary_sha256':digest(compact(self.vocabulary)),'idf_sha256':digest(self.idf.tobytes()),
   'bank_ids_sha256':digest(compact([r['source_id'] for r in bank]))}
 @staticmethod
 def terms(text):
  words=re.findall(r'(?u)\b\w+\b',text.lower())
  return collections.Counter(['u:'+w for w in words]+['b:'+a+' '+b for a,b in zip(words,words[1:])])
 def encode_counts(self,rows):
  values=[];indices=[];indptr=[0]
  for row in rows:
   pairs=sorted((self.position[w],v) for w,v in row.items() if w in self.position)
   weights=np.array([count*self.idf[i] for i,count in pairs],dtype=np.float64)
   norm=float(np.linalg.norm(weights))
   if norm:weights/=norm
   values.extend(weights);indices.extend(i for i,v in pairs);indptr.append(len(values))
  return csr_matrix((values,indices,indptr),shape=(len(rows),len(self.vocabulary)),dtype=np.float64)
 def scores(self,target):return (self.matrix@self.encode_counts([self.terms(feature_string(target))]).T).toarray().ravel()
