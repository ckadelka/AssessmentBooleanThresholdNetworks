#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:06:02 2019

@author: ckadelka
"""

import numpy as np
import os 


folder = 'update_rules_models_in_literature_we_randomly_come_across/'
pmid = '26564978'

text='''1. AKT*=(CARMA1) or (CDC42+RAC) or (COT) or (GRB7) or (IKK_ALPHA and IKK_BETA) or (PAK) or (PDK1) or (PKC)
2. AP1*=( not GSK3_BETA) or (ATF2 and JUN and not GSK3_BETA) or (CRE and ATF2 and JUN and not GSK3_BETA) or (CRE and JUN and not GSK3_BETA) or (FOS and JUN and not GSK3_BETA) or (JUN and not GSK3_BETA) or (NUC_ERK1_2 and FOS and JUN and not GSK3_BETA) or (NUC_JNK and FOS and JUN) or (NUC_P38 and FOS and JUN and not GSK3_BETA)
3. ASK1*=(TRAF2)
4. ATF2*=(NUC_P38)
5. BAD*=( not AKT) or (JNK and not AKT)
6. BCL10*=(CARMA1 and PKC_THETA and not IKK_ALPHA and not IKK_BETA and not IKK_GAMMA)
7. BCL2*=( not JNK) or (ETS) or (NUC_CREB)
8. BCLX*=( not BAD) or ( not JNK) or (ETS) or (NUC_NFKB)
9. C3G*=(CRK_L)
10. CABIN1*=( not CAMK4)
11. CALCINEURIN*=( not CABIN1 and CAM) or (CAM and not CALCIPRESSIN and not CABIN1)
12. CALCIUM_IN*=(CRAC and CALCIUM_OUT)
13. CAM*=(CALCIUM_IN) or (VAV and CALCIUM_IN)
14. CAMK4*=(CAM)
15. CARMA1*=(PKC_THETA)
16. CCL19*=(NUC_NFKB)
17. CD2*=(FYN) or (LCK)
18. CD3*=(LCK)
19. CD4*=(LCK)
20. CD8*=(LCK)
21. CDC25*=(NUC_MYC)
22. CDC42*=(PAK) or (RAS) or (VAV)
23. CDC42+RAC*=(VAV and not RAC_GAP)
24. CDK_4*=(NUC_MYC)
25. COT*=(RIP1) or (TRAF2)
26. CRAC*=(IP3)
27. CREB*=(RSK) or (not GSK3_BETA)
28. CRK_L*=(TYK2)
29. CYCLIN_A*=(AP1 and not GSK3_BETA) or (NUC_CREB) or (NUC_MYC)
30. CYCLIN_D1*=(AP1 and not GSK3_BETA) or (ETS) or (NUC_CREB) or (NUC_MYC) or (NUC_NFKB)
31. CYCLIN_D2*=(NUC_MYC)
32. CYCLIN_E*=(NUC_MYC)
33. DAG*=(PIP2)
34. ELK1*=(NUC_ERK1_2)
35. ERK1_2*=(MEK1_2 and not MKP)
36. ETS*=(NUC_ERK1_2)
37. FASL*=(ETS) or (NUC_NFKB)
38. FKHR*=( not AKT)
39. FOS*=(ELK1) or (ETS) or (JUN) or (NUC_P38)
40. FYN*=( not PAG+CSK) or (CD45 and not PAG+CSK and not CBL)
41. GAB1*=(ERK1_2) or (SHC)
42. GCKR*=(TRAF2)
43. GLK*=(TRAF2)
44. GM_CSF*=(ETS and NUC_NFKB) or (ETS and NUC_NFKB and AP1)
45. GRB2+SOS*=(B7_1 and CD28) or (B7_2 and CD28) or (RAS_GRP)
46. GSK3_BETA*=( not AKT) or (IFN_GAMMA)
47. HBEGF*=(ETS)
48. HPK1*=(LAT)
49. IFN_GAMMA*=(NUC_NFAT and AP1)
50. IKB_ALPHA*=( not IKK_BETA)
51. IKB_BETA*=( not IKK_ALPHA and not IKK_BETA) or ( not IKK_ALPHA and not IKK_BETA and not IKK_GAMMA)
52. IKK_ALPHA*=(NIK) or (TRAF2)
53. IKK_BETA*=(BCL10) or (IKK_ALPHA) or (PKC_THETA) or (TRAF2)
54. IKK_GAMMA*=(BCL10 and MALT1 and CARMA1) or (CARMA1 and MALT1 and BCL10 and IKK_ALPHA and IKK_BETA) or (IKK_ALPHA and IKK_BETA) or (RIP1) or (TAK1+TAB and RIP1) or (TRAF6 and MALT1)
55. IL1*=(NUC_NFKB)
56. IL10*=(AP1 and CREB and not GSK3_BETA) or (NUC_NFAT)
57. IL12*=(ETS and NUC_NFKB)
58. IL13*=(NUC_NFAT and AP1)
59. IL2*=(AP1 and NUC_NFAT and not GSK3_BETA) or (ETS and NUC_NFKB) or (NUC_NFAT and AP1) or (NUC_NFKB)
60. IL2R*=(NUC_NFKB)
61. IL3*=(ETS and NUC_NFKB)
62. IL4*=(AP1 and NUC_NFAT) or (AP1 and NUC_NFAT and not GSK3_BETA)
63. IL5*=(AP1 and not GSK3_BETA)
64. IL6*=(AP1 and CREB and not GSK3_BETA) or (NUC_NFKB)
65. IL8*=(NUC_NFKB)
66. IL9*=(AP1) or (NUC_NFAT) or (NUC_NFKB)
67. IP3*=(PIP2)
68. ITK*=(CD2) or (LCK)
69. JAK*=( not SOCS3) or (GRB2) or (IFN_ALPHA and IFNAR1_R2) or (IFNAR1_R2 and IFN_BETA) or (IFNAR1_R2 and IFN_OMEGA) or (SHC)
70. JAK2*=( not SHP2)
71. JNK*=(MKK) or (MKK4_7 and not MKP) or (MKK7) or (T3JAM)
72. JUN*=(FOS) or (NUC_JNK)
73. LAT*=(ITK) or (ZAP70)
74. LAT+GRB2+SOS1*=(LAT and GRB2 and SOS1)
75. LCK*=( not PAG+CSK and not LYP) or ( not PAG+CSK and CD4 and MHC_CLASS_II+AG) or (CD4 and MHC_CLASS_II+AG and not PAG+CSK and not LYP) or (CD45 and CD4 and MHC_CLASS_II+AG and CD28 and not CBL and not LYP and not PAG+CSK)
76. LYP*=( not CSK)
77. MALT1*=(CARMA1) or (PKC_THETA)
78. MEF2*=(CALCINEURIN and P300) or (CALCINEURIN and P300 and not CABIN1 and not HDAC) or (MEF2A and MEF2B and MEF2C and MEF2D)
79. MEK1_2*=(PAK and not MKP) or (RAF and not MKP) or (RAF1 and not MKP)
80. MEKK*=(CDC42+RAC) or (GCKR) or (HPK1) or (PAK)
81. MEKK1_4*=(CDC42+RAC) or (RAC1)
82. MEKK3*=(OSM)
83. MEKK4_7*=(CDC42+RAC)
84. MKK*=(ASK1) or (MEKK)
85. MKK3_6*=(MEKK1_4) or (MEKK3) or (MLK3)
86. MKK4_7*=(ASK1) or (COT and not MKP) or (MEKK4_7 and not MKP)
87. MKK7*=(MEKK) or (TAK1)
88. MLK2*=(PAK)
89. MLK3*=(CDC42 and not AKT) or (RAC)
90. NCK*=( not RAS) or (PKC and not RAS)
91. NCK+SOS*=(NCK and SOS)
92. NFAT*=(CALCINEURIN)
93. NFAT+P300+MEF2*=(NFAT and P300 and MEF2)
94. NFKB*=(OX40 and OX40L and PKC_THETA and TRAF2 and RIP1 and CARMA1 and MALT1 and BCL10 and IKK_ALPHA and IKK_BETA and IKK_GAMMA and not IKB_ALPHA and not IKB_BETA) or (not TRAF1) or ( not IKB_BETA and not IKB_ALPHA and NIK)
95. NIK*=(COT) or (TRAF2) or (TRAF5)
96. NUC_CREB*=(CREB) or (NUC_ERK1_2)
97. NUC_ERK1_2*=(ERK1_2)
98. NUC_JNK*=(JNK)
99. NUC_MYC*=(NUC_ERK1_2) or (NUC_NFKB)
100. NUC_NFAT*=(NFAT) or (NFAT and not GSK3_BETA)
101. NUC_NFKB*=(NFKB) or (NFKB and IL1)
102. NUC_P38*=(P38)
103. NUR_77*=(NFAT+P300+MEF2)
104. OSM*=(RAC1)
105. P15*=(NUC_MYC)
106. P21*=(AKT) or (NUC_MYC)
107. P21RAS*=(JAK2) or (LAT+GRB2+SOS1)
108. P38*=(MKK3_6)
109. P70*=(PDK1)
110. PAG+CSK*=(PAG and CSK and FYN and not CD45) or (PAG and CSK and LCK and not CD45)
111. PAK*=(ERK1_2 and not PIP) or (GRB2) or (NCK and not PIP)
112. PDGF*=(ETS)
113. PDGFRB*=(NUC_MYC)
114. PDK1*=(CARMA1) or (PIP3)
115. PI3K*=(B7_1 and CD28) or (B7_2 and CD28) or (GAB1) or (GRB2) or (ICOSL and ICOS) or (RAS) or (SHP2)
116. PIP2*=(PI3K) or (PLC_GAMMA)
117. PIP3*=(PIP2) or (PTEN)
118. PKC*=(JAK)
119. PKC_THETA*=(AKT) or (DAG) or (GLK) or (PDK1)
120. PLC_GAMMA*=(GAB1) or (GRB2) or (ITK) or (LAT) or (SHC) or (SHP2)
121. RAC*=(PAK and not RAC_GAP) or (RAS and not RAC_GAP) or (VAV and not RAC_GAP)
122. RAC_GAP*=(DAG)
123. RAC1*=(NCK) or (VAV)
124. RAF*=(PAK) or (PKC and not AKT) or (RAS)
125. RAF1*=(P21RAS)
126. RAP1*=(C3G)
127. RAS*=( not RAP1) or ( not RAS_GAP) or (GRB2+SOS) or (GRB7 and not RAS_GAP and not RAP1) or (NCK+SOS and not RAS_GAP and not RAP1) or (RAS_GRP) or (SHC+GRB2+SOS and not RAS_GAP and not RAP1) or (SHP1+GRB2+SOS and not RAS_GAP and not RAP1) or (SHP2+GRB2+GAB1+SOS and not RAS_GAP and not RAP1)
128. RAS_GAP*=(GRB2) or (NCK)
129. RAS_GRP*=(DAG and IP3) or (LAT)
130. RIP1*=(TRAF2)
131. RSK*=(ERK1_2)
132. SHC*=(IL2 and IL2R) or (PI3K) or (PKC)
133. SHC+GRB2+SOS*=(SHC and GRB2 and SOS)
134. SHP1*=( not ERK1_2) or (B7_1 and CTLA4) or (B7_2 and CTLA4) or (PDL and PD1)
135. SHP1+GRB2+SOS*=(SHP1 and GRB2 and SOS)
136. SHP2*=( not LCK) or (B7_1 and CTLA4) or (B7_2 and CTLA4) or (ERK1_2) or (SHC)
137. SHP2+GRB2+GAB1+SOS*=(SHP2 and GRB2 and GAB1 and SOS)
138. SLP76*=(ITK)
139. SOCS3*=(CRK_L) or (NCK)
140. SOS1*=(ERK1_2)
141. STAT1*=(PKC)
142. STAT3*=(PKC)
143. STAT5*=( not SHP2) or (JAK2) or (P38) or (PAK)
144. T3JAM*=(TRAF3)
145. TAK1*=(BCL10)
146. TAK1+TAB*=(RIP1)
147. TCR+CD3*=(TCR and CD3)
148. TGF_BETA*=(AP1 and not GSK3_BETA)
149. TNF_ALPHA*=(AP1 and not GSK3_BETA)
150. TRADD*=(TNF_ALPHA and TNFR) or (TNF_BETA and TNFR)
151. TRAF1*=(TNFSF9 and TNFRSF9) or (NUC_NFKB)
152. TRAF2*=(CD70 and CD27) or (LIGHT and LTBR) or (OX40L and OX40) or (TNFSF9 and TNFRSF9) or (TRADD)
153. TRAF3*=(CD70 and CD27) or (LIGHT and LTBR) or (TNFSF9 and TNFRSF9)
154. TRAF5*=(CD70 and CD27) or (LIGHT and LTBR)
155. TRAF6*=(MALT1)
156. TYK2*=(IFNAR1_R2 and IFN_ALPHA) or (IFNAR1_R2 and IFN_BETA) or (IFNAR1_R2 and IFN_OMEGA)
157. VAV*=(JAK) or (LAT and GADS and SLP76)
158. VEGF*=(NUC_NFKB)
159. WASP*=(NCK)
160. ZAP70*=(ABL and TCR+CD3 and MHC_CLASS_II+AG and not SHP1) or (LCK and not SHP1) or (LCK and TCR+CD3 and not LYP and FYN and ABL and VAV and not SHP1 and MHC_CLASS_II+AG) or (TCR+CD3 and MHC_CLASS_II+AG and FYN and not SHP1)
161. CTLA4*=(NUC_NFAT)
162. CD28*=(not TNF_ALPHA)
163. P53*=(ETS and NUC_P38)
164. PD1*=NUC_NFAT
165. GRB2*=PAK or SHC
166. GRB7*=SHC
167. PAG*=(not CD45 and LCK) or (not CD45 and FYN)'''

text = text.replace('*=',' = ').replace(')',' ) ').replace('(',' ( ').replace(' or ',' OR ').replace(' and ',' AND ').replace(' not ',' NOT ').replace('+','_').replace('  ',' ').replace('/','_')

g = open(folder+pmid+'.txt','w')
for line in text.split('\n'):
    g.write(line.split('. ')[1]+'\n')   
g.close()
