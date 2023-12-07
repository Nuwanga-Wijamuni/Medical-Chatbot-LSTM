{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from tensorflow.keras.layers import TextVectorization\n",
    "import re,string\n",
    "from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,LayerNormalization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "df = pd.read_excel(\"Allergy.xlsx\",header=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set display options to show full content without truncation\n",
    "pd.set_option('display.max_colwidth', None)\n",
    "NewData = df[['Patient','Doctor']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataframe size: 268\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Patient</th>\n",
       "      <th>Doctor</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Doctor, I've been feeling really strange lately. I had a reaction after eating something, and my throat felt like it was closing up. I'm really worried. Can you help me?</td>\n",
       "      <td>Of course, I'll do my best to help you. It sounds like you may have experienced an allergic reaction. Can you tell me more about what happened?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Well, I was at a restaurant and I had this dish with shrimp. Shortly after I finished eating, my lips started to swell, and I had difficulty breathing. I've never had this happen before, and it scared me.</td>\n",
       "      <td>I understand your concern. Based on your symptoms, it's possible that you had a severe allergic reaction called anaphylaxis. This is a serious condition that requires immediate medical attention. Have you experienced any other symptoms, such as hives, itching, or lightheadedness?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Yes, actually, I did notice some hives on my arms and chest, and I felt lightheaded and dizzy. I had no idea this could happen from eating shrimp. What exactly is anaphylaxis?</td>\n",
       "      <td>That's probably what's causing your reaction. Pork is a common allergen, and it can cause a variety of symptoms, including hives, difficulty breathing, and even anaphylaxis.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen, such as shrimp in your case. It can affect multiple systems in your body and can be life-threatening if not treated promptly. Common symptoms include swelling of the lips, face, or throat, difficulty breathing, hives or rash, and a drop in blood pressure leading to dizziness or fainting.</td>\n",
       "      <td>That sounds really serious! I had no idea an allergic reaction could be so dangerous. What should I do if it happens again?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Good morning, Doctor. I've been experiencing some unusual symptoms, and I'm not sure what's going on. I noticed some skin issues and recently had a blood test done. Can you please take a look at the reports and help me understand what's happening?</td>\n",
       "      <td>Good morning. Of course, I'll be happy to assist you. Please hand me your reports, and let's discuss your symptoms in detail. What specific skin issues have you been experiencing?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Well, I've been having frequent rashes and hives on different parts of my body. They appear as red, itchy patches, and they come and go randomly. It's quite uncomfortable, and I'm not sure what triggers them.</td>\n",
       "      <td>I see. Skin rashes and hives can be indicative of an allergic reaction. It's important to identify the underlying cause. Now, let's take a look at your blood test results. Could you please pass them to me?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Here are the reports, Doctor. I hope they can provide some insight into my condition.</td>\n",
       "      <td>Thank you. Let me review these reports.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Doctor, I had a really scary experience after eating pork yesterday. My face swelled up, and I had difficulty breathing. I think it might have been an allergic reaction. Can you help me understand what happened?</td>\n",
       "      <td>I'm sorry to hear about your distressing experience. Allergic reactions can indeed occur after consuming certain foods. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the facial swelling and difficulty breathing?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Yes, I also had hives on my body, and I felt quite dizzy. It was really frightening, and I had no idea that this could happen from eating pork. Is it possible that I have an allergy to pork?</td>\n",
       "      <td>It's possible that you have developed an allergy to pork. Allergic reactions can vary from person to person, and some individuals can be allergic to specific types of meat. To better understand your condition, it would be helpful to review your skin and blood test reports. Can you please provide me with those?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Certainly, Doctor. Here are the reports. I hope they can shed some light on what's happening.</td>\n",
       "      <td>Thank you. Let me take a look at the report .Based on your blood test results, your IgE levels are elevated, indicating a possible allergic reaction. Additionally, your skin prick test shows a positive reaction to pork allergens, further suggesting an allergy to pork.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>What should I do?</td>\n",
       "      <td>I recommend that you avoid beef in the future. If you have any more symptoms, please come back and see me.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Doctor, I had a scary experience after drinking a glass of milk yesterday. My throat started to swell, and I had trouble breathing. I suspect it might have been an allergic reaction. Can you please help me understand what happened?</td>\n",
       "      <td>I'm sorry to hear that you had such a distressing experience. Allergic reactions can occur after consuming certain foods, including milk. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the throat swelling and difficulty breathing?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>Yes, I also had hives on my skin, and I felt quite nauseous and lightheaded. It was really frightening, and I had no idea that drinking milk could cause such a reaction. Is it possible that I have an allergy to milk?</td>\n",
       "      <td>It's possible that you have developed an allergy to milk. Allergic reactions to milk are quite common. To better understand your condition, it would be helpful to review your skin and blood test reports. Could you please provide me with those?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>Certainly, Doctor. Here are the reports. I hope they can provide some insight into what's happening.</td>\n",
       "      <td>Thank you. Let me examine the reports and skin.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>Doctor, something really alarming happened after I ate some chocolate. My whole body started itching, and I had trouble breathing. I suspect it might have been an allergic reaction. Can you please help me understand what's going on?</td>\n",
       "      <td>I'm sorry to hear that you had such a distressing experience. Allergic reactions can occur after consuming certain foods, including chocolate. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the itching and difficulty breathing?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>Yes, I also had hives all over my body, and my face started to swell. I felt really anxious and had a rapid heartbeat. It was really scary, and I had no idea that chocolate could cause such a reaction. Could it be an allergy?</td>\n",
       "      <td>It's possible that you have developed an allergy to chocolate. Allergic reactions to chocolate can happen, although they are relatively rare. To better understand your condition, I would like to examine your skin and review your blood test reports. Could you please show me any visible skin reactions and provide me with the reports?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>Certainly, Doctor. Here are the reports, and you can see some red rashes on my arms and chest.</td>\n",
       "      <td>Thank you. Let me examine your skin and review the reports</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>Doctor, I had a frightening experience after eating beef. I developed a rash all over my body, and I had difficulty breathing. I suspect it might have been an allergic reaction. Can you please help me understand what's happening?</td>\n",
       "      <td>I'm sorry to hear about your distressing experience. Allergic reactions can indeed occur after consuming certain foods, including beef. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the rash and difficulty breathing?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>Yes, I also experienced severe itching and my face started to swell. I felt really anxious and had a tightness in my chest. It was really scary, and I had no idea that beef could cause such a reaction. Could it be an allergy?</td>\n",
       "      <td>It's possible that you have developed an allergy to beef. Allergic reactions to beef can occur, although they are relatively rare. To better understand your condition, I would like to examine your skin and review your blood test reports. Could you please show me any visible skin reactions and provide me with the reports?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>Certainly, Doctor. Here are the reports, and you can see the red rashes on my legs and arms.</td>\n",
       "      <td>Thank you. Let me examine your skin and review the reports.</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                                                                                                                                                                                                                                                                                                                                                                    Patient  \\\n",
       "0                                                                                                                                                                                                                                                 Doctor, I've been feeling really strange lately. I had a reaction after eating something, and my throat felt like it was closing up. I'm really worried. Can you help me?   \n",
       "1                                                                                                                                                                                                              Well, I was at a restaurant and I had this dish with shrimp. Shortly after I finished eating, my lips started to swell, and I had difficulty breathing. I've never had this happen before, and it scared me.   \n",
       "2                                                                                                                                                                                                                                           Yes, actually, I did notice some hives on my arms and chest, and I felt lightheaded and dizzy. I had no idea this could happen from eating shrimp. What exactly is anaphylaxis?   \n",
       "3   Anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen, such as shrimp in your case. It can affect multiple systems in your body and can be life-threatening if not treated promptly. Common symptoms include swelling of the lips, face, or throat, difficulty breathing, hives or rash, and a drop in blood pressure leading to dizziness or fainting.   \n",
       "4                                                                                                                                                                   Good morning, Doctor. I've been experiencing some unusual symptoms, and I'm not sure what's going on. I noticed some skin issues and recently had a blood test done. Can you please take a look at the reports and help me understand what's happening?   \n",
       "5                                                                                                                                                                                                          Well, I've been having frequent rashes and hives on different parts of my body. They appear as red, itchy patches, and they come and go randomly. It's quite uncomfortable, and I'm not sure what triggers them.   \n",
       "6                                                                                                                                                                                                                                                                                                                                     Here are the reports, Doctor. I hope they can provide some insight into my condition.   \n",
       "7                                                                                                                                                                                                       Doctor, I had a really scary experience after eating pork yesterday. My face swelled up, and I had difficulty breathing. I think it might have been an allergic reaction. Can you help me understand what happened?   \n",
       "8                                                                                                                                                                                                                            Yes, I also had hives on my body, and I felt quite dizzy. It was really frightening, and I had no idea that this could happen from eating pork. Is it possible that I have an allergy to pork?   \n",
       "9                                                                                                                                                                                                                                                                                                                             Certainly, Doctor. Here are the reports. I hope they can shed some light on what's happening.   \n",
       "10                                                                                                                                                                                                                                                                                                                                                                                                        What should I do?   \n",
       "11                                                                                                                                                                                  Doctor, I had a scary experience after drinking a glass of milk yesterday. My throat started to swell, and I had trouble breathing. I suspect it might have been an allergic reaction. Can you please help me understand what happened?   \n",
       "12                                                                                                                                                                                                 Yes, I also had hives on my skin, and I felt quite nauseous and lightheaded. It was really frightening, and I had no idea that drinking milk could cause such a reaction. Is it possible that I have an allergy to milk?   \n",
       "13                                                                                                                                                                                                                                                                                                                     Certainly, Doctor. Here are the reports. I hope they can provide some insight into what's happening.   \n",
       "14                                                                                                                                                                                 Doctor, something really alarming happened after I ate some chocolate. My whole body started itching, and I had trouble breathing. I suspect it might have been an allergic reaction. Can you please help me understand what's going on?   \n",
       "15                                                                                                                                                                                        Yes, I also had hives all over my body, and my face started to swell. I felt really anxious and had a rapid heartbeat. It was really scary, and I had no idea that chocolate could cause such a reaction. Could it be an allergy?   \n",
       "16                                                                                                                                                                                                                                                                                                                           Certainly, Doctor. Here are the reports, and you can see some red rashes on my arms and chest.   \n",
       "17                                                                                                                                                                                    Doctor, I had a frightening experience after eating beef. I developed a rash all over my body, and I had difficulty breathing. I suspect it might have been an allergic reaction. Can you please help me understand what's happening?   \n",
       "18                                                                                                                                                                                        Yes, I also experienced severe itching and my face started to swell. I felt really anxious and had a tightness in my chest. It was really scary, and I had no idea that beef could cause such a reaction. Could it be an allergy?   \n",
       "19                                                                                                                                                                                                                                                                                                                             Certainly, Doctor. Here are the reports, and you can see the red rashes on my legs and arms.   \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                                                           Doctor  \n",
       "0                                                                                                                                                                                                 Of course, I'll do my best to help you. It sounds like you may have experienced an allergic reaction. Can you tell me more about what happened?  \n",
       "1                                                        I understand your concern. Based on your symptoms, it's possible that you had a severe allergic reaction called anaphylaxis. This is a serious condition that requires immediate medical attention. Have you experienced any other symptoms, such as hives, itching, or lightheadedness?  \n",
       "2                                                                                                                                                                   That's probably what's causing your reaction. Pork is a common allergen, and it can cause a variety of symptoms, including hives, difficulty breathing, and even anaphylaxis.  \n",
       "3                                                                                                                                                                                                                     That sounds really serious! I had no idea an allergic reaction could be so dangerous. What should I do if it happens again?  \n",
       "4                                                                                                                                                             Good morning. Of course, I'll be happy to assist you. Please hand me your reports, and let's discuss your symptoms in detail. What specific skin issues have you been experiencing?  \n",
       "5                                                                                                                                   I see. Skin rashes and hives can be indicative of an allergic reaction. It's important to identify the underlying cause. Now, let's take a look at your blood test results. Could you please pass them to me?  \n",
       "6                                                                                                                                                                                                                                                                                                        Thank you. Let me review these reports.   \n",
       "7                                                                                 I'm sorry to hear about your distressing experience. Allergic reactions can indeed occur after consuming certain foods. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the facial swelling and difficulty breathing?  \n",
       "8                         It's possible that you have developed an allergy to pork. Allergic reactions can vary from person to person, and some individuals can be allergic to specific types of meat. To better understand your condition, it would be helpful to review your skin and blood test reports. Can you please provide me with those?  \n",
       "9                                                                    Thank you. Let me take a look at the report .Based on your blood test results, your IgE levels are elevated, indicating a possible allergic reaction. Additionally, your skin prick test shows a positive reaction to pork allergens, further suggesting an allergy to pork.  \n",
       "10                                                                                                                                                                                                                                     I recommend that you avoid beef in the future. If you have any more symptoms, please come back and see me.  \n",
       "11                                                              I'm sorry to hear that you had such a distressing experience. Allergic reactions can occur after consuming certain foods, including milk. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the throat swelling and difficulty breathing?  \n",
       "12                                                                                            It's possible that you have developed an allergy to milk. Allergic reactions to milk are quite common. To better understand your condition, it would be helpful to review your skin and blood test reports. Could you please provide me with those?  \n",
       "13                                                                                                                                                                                                                                                                                                Thank you. Let me examine the reports and skin.  \n",
       "14                                                                 I'm sorry to hear that you had such a distressing experience. Allergic reactions can occur after consuming certain foods, including chocolate. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the itching and difficulty breathing?  \n",
       "15  It's possible that you have developed an allergy to chocolate. Allergic reactions to chocolate can happen, although they are relatively rare. To better understand your condition, I would like to examine your skin and review your blood test reports. Could you please show me any visible skin reactions and provide me with the reports?  \n",
       "16                                                                                                                                                                                                                                                                                     Thank you. Let me examine your skin and review the reports  \n",
       "17                                                                           I'm sorry to hear about your distressing experience. Allergic reactions can indeed occur after consuming certain foods, including beef. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the rash and difficulty breathing?  \n",
       "18             It's possible that you have developed an allergy to beef. Allergic reactions to beef can occur, although they are relatively rare. To better understand your condition, I would like to examine your skin and review your blood test reports. Could you please show me any visible skin reactions and provide me with the reports?  \n",
       "19                                                                                                                                                                                                                                                                                    Thank you. Let me examine your skin and review the reports.  "
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(f'Dataframe size: {len(NewData)}')\n",
    "NewData.head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 268 entries, 0 to 267\n",
      "Data columns (total 2 columns):\n",
      " #   Column   Non-Null Count  Dtype \n",
      "---  ------   --------------  ----- \n",
      " 0   Patient  246 non-null    object\n",
      " 1   Doctor   258 non-null    object\n",
      "dtypes: object(2)\n",
      "memory usage: 4.3+ KB\n"
     ]
    }
   ],
   "source": [
    "NewData.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Nuwanga Wijamuni\\AppData\\Local\\Temp\\ipykernel_23328\\1690811942.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  NewData['question Tokens'] = NewData['Patient'].apply(lambda x: len(str(x).split()))\n"
     ]
    },
    {
     "data": {
      "image/png": "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iIiIeE+AxeDfQxoyslWgx3KHCx745jQf7sv1WC4iIiIiIiLe162RH3e3CyLcWn62MTnXwfu/5HEwy+7FyEREPKsVCzrPmzePTZs2AbB7924APvzwQ9avXw9A3759GTt2LAATJ05k2bJlvPHGG+zYsYNLL72Un376iTVr1tC9e3fGjx9fM29CRERExMv8TAb/HNSgeDnVA2VnL7qAxzZkkF3k4pHOId4PUERERERERMpoGmTm3vbBLEsuYH+m52Rint3Fx0n59I6yMrCJFbOpgmmQIiLVqFYkGjdt2sSCBQtKHdu8eTObN28u+d2daAwODiYhIYFXX32VJUuW8O233xIdHc2jjz7K008/TWCg57v6RUREROois8ngrQER+BsO/rW/0GOdZ77L5Fiegxd6hmEyNDgVEalqH330EZs2bWL79u3s3r0bm83GzJkzufvuuz3Wz8rK4tVXX2Xx4sX8+uuvREdHM2LECJ5++mlCQnRjiIiISH0QYDEY2SqA708W8XVqIc5ytmXc8quNg9l2bogNoHGg2btBiohQSxKNs2fPZvbs2eddPzw8nKlTpzJ16tRqjEpERESkdjAZBq/0CMaRn8O8FD+PdWbsyiE118GsgQ3wr2hDEBERqbSXX36Z5ORkGjVqRHR0NMnJyeXWzc3NJT4+np07dzJkyBBGjx7Njh07mDFjBhs2bGDZsmUEBHjeg1dERETqFsMw6NnYSrMgM18eyieryHO28dd8Jx/sy2NwM396RPph6AZSEfGiOrVHo4iIiIh4ZhgGj7Ys4umu5a/u8NnBfEb/9ySZNqcXIxMRqftmzJjBjh07SExM5He/+12Fdd9880127tzJpEmTWLRoEVOmTGHRokVMmjSJH374gVmzZnkpahEREfEVzYLN3HdJMO3Cyp835HDB6qOFfJyUT7bGdCLiRUo0ioiIiNQThgF/6BLE1F7h5db59riN65adIDXX4cXIRETqtsGDBxMbG3vOei6Xiw8//JCQkBAmT55cqmzy5MmEhIQwb9686gpTREREfFiAxWBk6wCGNPOnou0YD2U7mPtLLntOF3kvOBGp15RoFBEREalnxncO4b1BDbCWcyW4+7SdYQknNDAVEfGyxMREjh07Ru/evQkODi5VFhwcTO/evTl06BApKSk1FKGIiIjUJMMwuCLKyth2QUQGlP/VfoEDFh8uYPGh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z+zs7HLvDBURERERERGpSRrTioiIiIhIbVLnEo1t2rQBIDExsUxZWloaOTk5xMXFeTssERERERERkXPSmFZERERERGqTOpdo7N+/PwBr1qwpU7Z69epSdURERERERER8ica0IiIiIiJSmxgZGRmumg6iKtntdnr27MmxY8dYuXIl3bp1AyAzM5OhQ4dy5MgRtm7dSsuWLWs4UhEREREREZHSNKYVEREREZHapM4lGgHWrVvHLbfcQkBAAKNGjSIkJITFixeTnJzMSy+9xGOPPVbTIYqIiIiIiIh4pDGtiIiIiIjUFnVu6VSAK6+8khUrVtC7d28+//xz5s6dS1RUFHPnzq2xAdkPP/zArbfeSmxsLM2aNePqq6/m888/r5FY5H8++ugjJk2axODBg4mKiiIiIoL58+eXWz8rK4tnnnmGLl26EBUVRdeuXXnuuefIycnxYtT1V2pqKrNmzWLkyJF06dKFxo0b0759e8aMGcO2bds8PkZ9VrMKCgp45plnuO666+jQoQPR0dG0b9+e4cOH8+9//5uioqIyj1Gf+aY33niDiIgIIiIi2Lp1a5ly9VvN6tq1a0n/nP0nPj6+TP3CwkKmTZtG9+7diY6OpkOHDkycOJETJ07UQPT125IlSxgxYgStW7cmOjqabt26cf/995OSklKqnj5jNWv+/Pnlfsbcf2666aZSj1GfycXwxTGtJxrn1i8aj8m5aMwgurYVAJfLxeLFi7nhhhu45JJLaNq0KT179mTSpEkcOnSoTH2dD3VDdX/P7nQ6eeedd+jXrx9NmjShTZs23H///R7PKfG+Ojmj0dfoblTf1bVrV5KTk2nUqBFBQUEkJyczc+ZM7r777jJ1c3Nzufbaa9m5cydDhgyhW7du7NixgzVr1tC9e3eWLVtGQEBADbyL+mPKlCm88cYbtG7dmgEDBhAZGUliYiIJCQm4XC7++c9/MmrUqJL66rOad+rUKTp37kz37t1p27YtkZGRZGRksHLlSpKTkxkyZAiffvopJlPxfS/qM9+0e/durrrqKiwWC7m5uaxcuZIrrriipFz9VvO6du1KZmYm48ePL1MWGxtb6v81p9PJrbfeyurVq7niiivo378/iYmJLF26lJYtW7Jq1SoiIyO9GX695HK5+MMf/sD7779P69atGTp0KCEhIRw7dowNGzbw7rvv0rdvX0CfMV+w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JZdcUqq+zoe6obq/Z3c6ncyZM4cPPvigZEnmwYMH89xzz5UkM6XmKNEoIiIiIiIiIiIiIiIiIpWmPRpFREREREREREREREREpNKUaBQRERERERERERERERGRSlOiUUREREREREREREREREQqTYlGEREREREREREREREREak0JRpFREREREREREREREREpNKUaBQRERERERERERERERGRSlOiUUREREREREREREREREQqTYlGEREREREREREREREREak0JRpFRKTOmD9/PhEREcTHx9d0KD4hPj6eiIgI5s+fX9OhiIiIiIiIiJSYOnUqERERjB8/vqZDERGRi2Sp6QBERETOx9KlS9m5cycDBgxg4MCBNR3OBZs6dSrTpk2r9ONatGjBzp07qyEiERERERERkbJmzZpFZmYmd911Fy1btqzpcERExEcp0SgiIrVCQkICCxYsACg30RgWFka7du2IiYnxZmiVEhMTQ58+fcocT0lJISUlBX9/fy6//PIy5dHR0d4IT0RERERERASA2bNnk5yczIABA5RoFBGRcinRKCIidcaNN97IjTfeWNNhVGjMmDGMGTOmzHH3TMeoqChWrFhRA5GJiIiIiIiIiIiIVI72aBQRERERERERERERERGRSlOiUUSkDjl58iRPPvkknTt3Jjo6mq5duzJ58mROnz5d7kbr3377LREREXTt2rXc5x0/fjwRERFMnTrVY3lGRgbTpk1j0KBBxMbGEh0dTc+ePXn22Wc5ceKEx8dkZWXxyiuvMGDAAJo3b07jxo255JJLGDx4MH/+859JSkoC4PDhw0RERJQsmzpt2jQiIiJK/pwZ9/z584mIiCA+Pt7ja+bm5vL6668zePBgWrRoQdOmTbniiit45plnOH78+Dnfe35+Pq+88go9e/YkOjqaNm3acN9995GYmFhu21WV9PR0XnzxRfr27UuzZs1o3rw5/fr145VXXiEzM7PSz3f8+HEGDBhAREQEt9xyCzk5OSVl+fn5zJo1i+HDh9OyZUuioqLo1q0bkyZN4tChQx6fLz4+noiICObPn8/p06f505/+RNeuXYmKiqJjx448/vjjpKWleXzs+Z4LIiIiIiJSvx04cIDXX3+dG264gS5duhAdHU1sbCzDhg3j7bffxmazeXzcxYzrCgsLeeuttxg6dCixsbFERkbStm1b+vXrx5NPPsn27dtL6r788stERETw5JNPlnmed955p2Qcu3z58jLlo0aNIiIignnz5pUpO378OH/5y1/o27cvzZs3p1mzZvTr149XX32V7Oxsj3G7X+vw4cN8//33jB07lvbt29OwYcNyx/Zu7rF1cnIyULx60Jnj8LO/VygqKuK9997j2muvpWXLlkRHR3PppZcyceLECxrPFRUV8eCDDxIREcGll17KgQMHSpUvXryY22+/nXbt2tG4cWPatWvHXXfdxYYNGzw+35nfhzgcDmbOnEm/fv1o2rQpLVu25Pbbby/Vj2dyOp3MmzeP66+/nlatWhEZGUlcXBy9e/dmwoQJrFu3rtLvT0SkrtHSqSIidcThw4eJj48nJSUFk8lEhw4dcLlc/POf/2TlypUMHz68Wl53586d3H777aSmpmKxWGjRogWBgYEcOHCAt956i08//ZRFixbRqVOnksdkZ2dzzTXX8Msvv2AYBq1btyYiIoITJ07w888/s337di655BLi4uIICAigT58+JCYmcuLECWJiYkrtwXi+exceO3aMkSNHsnfvXgzDoH379vj7+7Nnzx5mzZrFwoUL+fjjj+nZs6fHx7tj/vnnn2nfvj1xcXHs37+fzz//nG+++Ya1a9cSGxt7cY1Zjr179zJq1ChSU1Mxm80lfbt37152797NwoUL+eKLL4iLizuv59u3bx+33HILycnJ3H777bz11lv4+fkBkJyczK233srevXsxmUw0a9aMFi1akJSUxPvvv89nn33Gf/7zn3L3yUxNTWXgwIEcP368pI2TkpKYN28e69atY926dYSFhZXUr8y5ICIiIiIi9duLL77I4sWLCQkJISoqis6dO3PixAm+++47vvvuO5YsWcLnn3+O1Wr1+PjKjuscDgejRo0qSWDFxsbStm1bTp8+TVJSErt37yYiIoLLLrsMgIEDB/Laa695TD598803JX9ft24d1113XcnvRUVFbN68GYArr7yyzOPGjBlDVlYWVqu1ZK/EX375hVdffZXPPvuMxYsX07RpU4/vefHixbzwwgsEBATQtm1bwsLCMAyjwnaOioqiT58+/PjjjxQWFtKpU6dS47i2bduWatPbbruNTZs2AdCqVSsiIiLYt28fH3zwAR9//DFz584t9X4rkp2dzZgxY1i7di3dunXjk08+KRn3FxYW8sADD7B48WIAIiMj6dixI8nJySxbtozly5fz4osv8thjj3l8bofDwa233sqaNWuIi4ujTZs27N+/n6+++op169aRkJBA9+7dSz3m4Ycf5uOPPwagadOmtG7dmuzsbFJSUvjll18oKioq02ciIvWNZjSKiNQRDz/8MCkpKXTs2JFt27axceNGNm3axObNmzGZTMydO7fKX/P06dPccccdpKamMm7cOPbu3cuPP/7Ixo0b2b9/P3fccQfHjx9n3Lhx2O32ksd9+OGH/PLLL3Tq1Int27fzww8/sGbNGnbu3ElycjLvv/8+HTp0AIoTiStWrODqq68G4O6772bFihUlfz744IPzivWBBx5g7969tGnThg0bNrBlyxbWrVvHzz//zJVXXkl6ejpjx44td3bgu+++i9ls5vvvv2fLli1s2rSJbdu20a5dO9LT03nllVcusjU9KywsZMyYMaSmptKzZ0+2b9/Ohg0b2LhxI99//z1dunThyJEjjB07FofDcc7n++6777j22mtJTk5m4sSJvP322yVJRpvNxp133snevXu5/vrr2b59O7t27WL9+vUcPHiQSZMmkZ2dzb333svp06c9Pv/f/vY32rdvz65du9i4cSPbtm3j66+/JioqikOHDvHWW2+Vql+Zc0FEREREROq322+/ndWrV5OcnFxq7PDdd99xxRVXsGHDBmbOnFnu4ys7rlu+fDkbNmygWbNmrF+/nh07drBmzRp+/PFHUlJS+OSTT+jTp09J/d69exMQEMC+fftKrZrjdDrZsGEDjRs3xs/Pr0wicuvWreTl5REbG0urVq1KjiclJXHPPfeQlZXFk08+SVJSElu3bmXr1q38/PPPXHPNNezfv5+HHnqo3Pc8ZcoUHn74YQ4cOMDatWvZtm0bEydOrLCdr7nmGlasWEFUVBRQvLLQmePwP/7xjyV1n376aTZt2kRkZCTLly9n+/btrF27lr179zJ69Gjy8/N54IEHOHz4cIWvCZCWlkZ8fDxr165l0KBBJCQklLq5+JlnnmHx4sV07NiRFStWcODAAdatW8fBgweZM2cOgYGB/OUvf2H9+vUen//zzz8vaYcffviB9evXs3v3bnr37k1+fj7PPvtsqfo7duzg448/JiwsjCVLlrBnzx6+/vprtm3bRkpKCgkJCeedQBURqcuUaBQRqQPcSUUoXo7lzNlfl1xyCbNmzaKoqKjKX3fmzJkcPXqU66+/njfffJPIyMiSsvDwcGbOnEm3bt3Yv38/S5YsKSnbv38/AGPGjCm5G9MtICCAESNG0KtXryqLc+PGjSUDjXfffbfU7MqoqCjmzZtHWFgYqampHpepATCZTLz//vul2rZVq1Y899xzAKxYsaLK4j3T559/zv79+7FarXzwwQe0aNGipKx169a8//77mM1mdu3axdKlSyt8rmXLlnHzzTeTkZHBq6++ygsvvFDqTtaFCxeya9cuLr/8cj744INSd/L6+/szZcoUrr32Wk6dOlVuO4WFhTF37lyaNGlScuzSSy/l8ccfB8q2k7fPBRERERERqb3i4+Pp0aNHmRl57du355133gEo2XbDk8qO69zjlZtvvpkuXbqUKrNYLFxzzTUlN8VC8RjmiiuuACiVTPzpp5/IzMxkyJAh9OjRg927d3Pq1KmScnfds2fGuZdGfeihh3j22WcJCQkpKWvSpAlz586lWbNmrFu3ju+//97jex40aBAvv/wyAQEBJccCAwPLa6JKOXz4MAsXLgTgtddeo2/fviVlYWFhvP3227Rs2ZKcnJwyN52e7cCBAwwbNowdO3Zw66238sknnxAaGlpSvn//fv71r38RFhbGRx99VCrBC3DbbbfxzDPP4HK5ePPNNz2+RlFREW+//XbJDFSARo0aMW3aNAA2bdpU6uZjd/8PHDiwzKo+hmHQv39/Ro4cWeH7EhGpD5RoFBGpA1auXAlAv3796NatW5nyPn36lFn+oyosWrQIgN/97ncey81mM9dffz1QepkYd7JsxYoVpfYGrC7//e9/Aejbt6/HdoiIiOCee+4pVfdsQ4YMoXXr1mWOu5NgGRkZ5c7yuxjueEaMGEHz5s3LlLdt27bkDsryYgd4//33GTNmDE6nk7lz5/Lwww+XqePuzzFjxpTMcjzbTTfdBJTuzzONHj2aiIiIMsfd7XTw4MFSx719LoiIiIiISO124sQJ3n77bR588EFGjBjBddddx7XXXssjjzwCFCeH8vPzPT62suM693hl7dq1nDx58rzicycLzxwzuf9+5ZVXMnDgQFwuV6lEpKdEY1FRUcnNpPfff7/H1woNDWXw4MFlXu9MY8aMOa+4L8Tq1atxOp3ExMSUjBXPZLFYSvZzrGi8um3bNoYPH87hw4eZMGECc+bMKbP87ZdffonT6eTqq68ud9sSdwzr16/3uOJP586d6devX5njl156Kf7+/rhcrlJjVnf/b9u2jUOHDpUbv4hIfac9GkVE6oB9+/YB0LFjx3LrdOjQgR9++KHKXjM3N7dkU/e//vWvvPbaax7r/frrrwAcPXq05Ng999zDzJkz+eabb+jQoQODBw+md+/eJQlRs9lcZXHC/+5CrKh93LMc3XXPduYeFGdyLyUDxXtJNGjQ4ELD9Mgdz5mzMM/WqVMnli5dWm7s//znP/nxxx8JCwtj/vz55e6vuGvXrpL67j0ozua+u/PM/jzTudopOzu71HFvnwsiIiIiIlJ7ffnll0yYMKHCmxRdLhenT5/2OGuvsuO6+Ph42rVrx549e+jcuTMDBw6kb9++9OrVi169euHv71/mua688kr++te/8u2335YcOzORGBsby/Tp01m3bh0jR44kPz+fbdu2lZS7JSYmkpeXB1CyQownycnJQPljtOrcisI9Bu3QoQMmk+f5LO6x7OHDh7HZbGUSiD/++CM33XQT+fn5vPzyyzz66KMen8c9XnVvB+KJy+UCID8/n/T0dBo3blyqvLz+NwyDxo0bk5KSUurcuuKKK+jfvz8bNmygR48e9O3bl379+tGrVy/69OlTaoapiEh9pkSjiEgd4L4QPvsi+kxnDpyqwpnLifz444/nrO8eILljWb16NdOmTSMhIYGlS5eW3KkZGRnJ+PHjmThxIhZL1fw35W6fitrAvdRneQPWoKAgj8fPHEy5BzVVqSpiT0xMBKBhw4Ye7951y8jIAGD37t3njOvM/jxTee109tJGbt4+F0REREREpHY6fPgwDz74IIWFhYwcOZKHHnqI9u3bExYWhsViwel00rBhQ4Bytw6p7LguMDCQ5cuXM23aNBYtWsSqVatYtWoVULw06NixY3nmmWdKPW+PHj0ICQnhyJEjHDp0iObNm7N582bi4uJo0aIFUVFRBAYGliQfN2/ejM1mo3379qW2oHCPz9x1zqWyY7SqUJnxqru+u4/cUlNTycvLIyAggK5du5b7PO72SElJISUl5ZyxeWqPitrCPWY9s/8Nw+Cjjz7ijTfeYMGCBaxfv75kW5bAwEBuueUWXnjhBRo1anTOeERE6jJ9ayciUge476I7ceJEuXXcMwvP5uli+myeLtCDg4NL/r59+/ZSG9afj1atWjF79mwcDgc7d+5k8+bN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JYW3336b559/niZNmtC0aVNsNhuHDh0iNzcXi8XCG2+8UTLb0s1kMtG/f38SEhJwuVxllkW1WCz069evJGF6drlbu3bt+M9//sO9997LBx98wIcffkibNm2IiIggJyenZNwExTMCq9odd9zB8uXLmTlzJgkJCTRt2hSTycTVV1/NH/7wBwCmTZvGwYMH2bRpE9dddx1xcXGEh4fzyy+/kJeXR2BgIO+++y4tW7Y8r9d03yj91FNPcf/99+NwOBg1ahQA06dPJz8/n48//pg77riDiIiIkmTo8ePHOXbsGAB33nlnlbz/U6dOMX36dKZPn07Dhg1p0aIFLpeLw4cPl4yNp0yZUrKtjIhIfaUZjSIidUSrVq1Yu3Yt999/P02aNGH//v1kZWXx+9//nq+//poGDRqU+9gePXqwYsUKhg8fjr+/PwcOHMBqtfLiiy/y0UcfVZjA6ty5Mxs3buSVV16hX79+ZGZm8uOPP5KcnEyrVq144IEH+OKLL0pmTULxAOjJJ5+kb9++uFwudu7cyaFDh2jZsiUPPPAAGzduLLlb1S0mJoZFixZxzTXX4HK52Lp1Kxs2bDjvfTiaNm3K6tWr+ctf/kK3bt1ISUlh3759tGzZkvHjx7Nx40Z69ux5Xs/lbR07dmTDhg088cQTtGvXjqSkJA4dOkSHDh2YPHky69atIy4u7ryfz2w2M2fOHO68805+/fVXbrzxxpLZizExMaxatYoZM2YwZMgQCgoK2L59O0lJSURHR3PPPffwn//8h1tuuaVK3tuFnAsiIiIiIlL/mEwmPvnkE/7whz/QsmVLUlNTSUlJYeDAgSxevLhkZl1Vuv/++3n22WcZNGgQfn5+7N27lwMHDhAVFcXdd9/N2rVrS5JgZzszeegpkXiucreBAweydetWnnnmGbp3705aWho//PADqampdOzYkccff5yvvvqK2NjYi3innt1888289dZb9OzZk1OnTrF582Y2bNhQaoZfaGgoixcv5rXXXqN3796cPHmSn3/+mUaNGjF27FjWr1/PddddV6nXfeCBB3jjjTdwOBw88MADJTc+W61W5syZwxdffMGoUaMICQlh9+7d7N69G4vFQnx8PDNmzODll1+ukvd/00038dJLLzF8+HBCQ0M5cOAAe/fuJTQ0lFGjRrFs2TIee+yxKnktEZHazMjIyHCdu5qIiNR2U6dOZdq0adx5553Mnj27psMRERERERERERERkVpOMxpFREREREREREREREREpNKUaBQRERERERERERERERGRSlOiUUREREREREREREREREQqTYlGEREREREREREREREREak0IyMjw1XTQYiIiIiIiIiIiIiIiIhI7aIZjSIiIiIiIiIiIiIiIiJSaUo0ioiIiIiIiIiIiIiIiEilKdEoIiIiIiIiIiIiIiIiIpWmRKOIiIiIiIiIiIiIiIiIVJoSjSIiIiIiIiIiIiIiIiJSaUo0ioiIiIiIiIiIiIiIiEilKdEoIiIiIiIiIiIiIiIiIpWmRKOIiIiIiIiIiIiIiIiIVJoSjSIiIiIiIiIiIiIiIiJSaf8fToGRTQmYNHsAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 2000x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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o7Lx3BIRERGRAwxMRERERA4wMBERERE5wMBERERE5AADExEREZEDDExEREREDjAwERERETnAwERERETkAAMTERERkQMMTEREREQOMDAREREROcDAREREROQAAxMRERGRAwxMRERERA4wMBERERE5wMBERERE5AADExEREZEDDExEREREDjAwERERETnAwERERETkAAMTERERkQMMTEREREQOMDAREREROcDAREREROQAAxMRERGRAwxMRERERA4wMBERERE5wMBERERE5AADExEREZEDDExEREREDjAwERERETnAwERERETkAAMTERERkQMMTEREREQOMDAREREROcDAREREROSAIgPThx9+iLlz5+Lqq69GQkICoqKisG7dOrvb19XV4cknn8Tll1+OhIQE9O/fH8888wwMBoPN7dvb2/Haa69h1KhR6NGjBzIzM3HfffehqKjIQ5+IiIiI5EyRgWnhwoVYs2YNiouLodfru922oaEBU6ZMwapVq9C7d288/PDDyM7OxooVK3DTTTehqampy2vmzp2L+fPnw2Qy4cEHH8SECROwYcMGXHPNNcjPz/fUxyIiIiKZUmRgWrFiBQ4ePIj8/Hz8/ve/73bb5cuX49ChQ5g7dy7Wr1+PBQsWYP369Zg7dy727duHVatWWW2/bds2rF27FqNGjcLWrVvx3HPP4fXXX8e6detQXV2Nxx9/3JMfjYiIiGRIkYHp6quvRkpKisPtTCYT3nnnHeh0ui5B5/HHH4dOp8PatWutHjf/91NPPQWNRmN5fOLEiRgzZgxyc3NRXFwswqcgIiIipVBkYHJWfn4+SktLMXz4cGi1WqvntFothg8fjqKiIpw9e9by+I4dO6DVajFixIgu7zdhwgQAwM6dOz1bOBEREcmKzwcmAMjIyLD5vPlx83YNDQ04f/48UlNToVarHW5PRERE/iFQ6gI8qa6uDgAQGRlp8/mIiAir7cz/a37c0fbdsTWZ3F8YjUar/yVhOI7u4xiKg+MoDjHGMSQkxO06/Pn4ZIuzY+rTgUlKJSUlaGtrk7oMSZWVlUldgk/gOLqPYygOjqM43BnH7Oxst/fP45M1Z8fUpwOTuSNUW1tr8/nOHSVHHSRHHaiOkpKSXCvWhxiNRpSVlUGv11tNnCfXcBzdxzEUB8dRHHIZR38+PrnDpwNTZmYmAKCgoMDm8+bH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EsClCdCpqalW7xkSEoKpU6di2LBhotWZl5eHHTt2ALgUfDp2uxISErB27VpERESgpKQEa9eutfkeAQEBWLNmjdXYpqWl4ZlnngEAj03U/uyzz3Dy5EloNBq8/fbbSE5OtjyXnp6ONWvWQK1W4/Dhw/jyyy+7fa+vvvoKN998M2pqarBkyRI899xzUKlUluc/+OADHD58GIMHD8bbb79t1TELDg7GggULcN1116GystLuOEVERGD16tXo0aOH5bGBAwfiz3/+M4Cu4+TtnwUif8PARNTB5s2bAQCjRo3CgAEDujw/YsQIDBkyRPT9rl+/HgDw+9//3ubzarUakydPBgBs3brV8rj5oL9p0yaruTOe8u233wIARo4caXMcoqKicM8991ht29n48eORnp7e5XHzwbympsZu18Ud5nqmTp2Knj17dnk+KysL119/vdW2tqxZswYzZsxAe3s7Vq9ejYceeqjLNua/zxkzZli6Tp3ddNNNAKz/Pju69dZbERUV1eVx8zgVFhZaPe7tnwUif8M5TEQdnDhxAgDQt29fu9tcdtll2Ldvn2j7bGhosEzG/cc//oEXX3zR5nbl5eUAgHPnzlkeu+eee7By5Ups3boVl112Ga6++moMHz7cEuzUarVodQL/62J0Nz7mrpN5286ysrJsPp6QkGD5//X19YiOjhZapk3mejp2xTrLycnBl19+abf2N998E7/88gsiIiKwbt06u/OPDh8+bNn+o48+srmN+ZRlx7/PjhyNU319vdXj3v5ZIPI3DExEHZh/M4+Pj7e7TccDuxg6zvX55ZdfHG5/8eJFq1q2bNmC559/Hhs3bsSXX35pOZ0UFxeH2bNn45FHHkFgoDj/1M3j090YmE8h2etyhIWF2Xw8IOB/DW+TySS0RLvEqN18FV9MTIzNLplZTU0NAODo0aMO6+r499mRvXHqeOqvI2//LBD5G/7LIepAp9MBgN1L+IH/dXo6Mx/IujvY2zo4arVay//fv38/0tLSnCnVIi0tDa+88gra2tpw6NAh7N69G5s3b0Zubi7+/ve/o66uDs8995xL72mPeXzsjQEAy7IC5m3lQozan332WWzcuBG5ubmYMmUKNmzYYPOKPq1Wi9raWnzxxRcYN26cCNU7x5s/C0T+hnOYiDro3bs3gEuXn9tj7zlz8KmoqLD72lOnTnV5LDIy0nJ5/5EjR5yutTO1Wo1BgwbhoYcewqeffooXXngBALB69WqrEGevQ+EM8/j8+uuvdrcxd1XM28qFuZ7uuj6Oag8JCcH777+Pa6+91rL8hK31lMyn/dz5+3SHsz8LROQ8BiaiDsyX3O/cudMyD6WjH3/80e78pfT0dKhUKjQ1NeHAgQNdnt+9e7fdA+jUqVMBXLpazplL2p0xfPhwAJfmunSc72I+1dPY2Ojye1577bUAgF27dtkch5qaGrz77rtW28qFuZ7//Oc/NucNFRQU4Ouvv7ba1pbg4GC8++67mDx5MoqLizFlypQuC27ecsstAC7NYbJ3ys2b7P0sEJHzGJiIOhg9erTl4PLAAw9YdQ9OnjyJhx9+2O5VT1FRURg5ciQA4K9//Suqqqoszx04cAAPPfSQ3dfOnTsXiYmJyMvLw4wZM7p0LUwmE/bt24e//vWvVkHlueeew1tvvdXlNFNNTQ1eeuklAJcmD3dc4NA892bXrl0wGo3dDUcXI0eOxJgxYwBcWo+pY6fpwoULuPfee1FXV4ekpCSb93mT0i233ILs7GwYjUb87ne/Q3FxseW5oqIi/O53v0NbWxsuv/xyuyucm2k0GqxduxY333wzzp07hylTplguGACAWbNmIScnB/n5+Zg2bZrN8P3rr79i4cKFlpDmLiE/C0TkPM5hIurktddew+TJk3H06FEMGTIEffv2hclkwq+//orU1FTce++9eP31122+9u9//zumTJmCXbt2IScnB1lZWWhsbER+fj4mTJiAYcOG2bxqKi4uDp988gnuuusufPXVV/jqq6+QlpaGuLg4XLx4EadPn0ZDQwMAWB3Mjx8/jpdeegmPPfYYevXqBb1ej4sXL6KgoADNzc3QarVYvny51b5uvvlm/OMf/8BPP/2EnJwcZGZmIjAwEHq93qlFOd944w3LSt+jRo1Cnz59oNFo8Ouvv6KlpQXR0dFYu3YtIiMjXRl2jzOHnGnTpuGnn37CoEGDrFb6bm9vR0pKCtauXevUFWWBgYFYvXo1HnzwQXzyySe44YYb8Pnnn6Nv374IDg7GRx99hLvuugu7d+/GmDFj0KtXL/To0QPNzc04c+aMZbL/ypUrRfl8Qn4WiMh57DARdZKWloYffvgB9913H3r06IGTJ0+irq4O999/P77//vtuL3e/4oorsGnTJkyaNAnBwcE4deoUNBoN/u///g8ffvhhtwfifv36IS8vD4sWLcKoUaNQW1uLX375BcXFxUhLS8Mf/vAH/Oc//7F0sQDgiSeewF/+8heMHDkSJpMJhw4dQlFREVJTU/GHP/wBeXl5GD16tNV+evXqhfXr12PixIkwmUz46aefsHPnTvz0009OjU9iYiK2bNmC//f//h8GDBiAs2fP4sSJE0hNTcXs2bORl5dn97YoUuvbty927tyJefPmITs7GwUFBSgqKsJll12Gxx9/HNu2bXPp1iFqtRqvv/467rzzTpSXl+PGG2+0dJN69eqF7777DitWrMD48ePR1NSE/fv3o6CgAHq9Hvfccw/ee+89TJ8+XZTPJuRngYicp6qpqeEMQCIXLF68GM8//zzuvPNOvPLKK1KXQ0REXsAOExEREZEDDExEREREDjAwERERETnAwERERETkACd9ExERETnADhMRERGRAwxMRERERA4wMBERERE5wMBERERE5AADExEREZEDDExEREREDjAwERERETnAwERERETkAAMTERERkQP/H5fi4yNKaO3DAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 600x600 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "NewData['question Tokens'] = NewData['Patient'].apply(lambda x: len(str(x).split()))\n",
    "NewData['answer tokens'] = NewData['Doctor'].apply(lambda x: len(str(x).split()))\n",
    "plt.style.use('fivethirtyeight')\n",
    "fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))\n",
    "sns.set_palette('Set2')\n",
    "\n",
    "sns.histplot(x=NewData['question Tokens'], data=NewData, kde=True, ax=ax[0])\n",
    "sns.histplot(x=NewData['answer tokens'], data=NewData, kde=True, ax=ax[1])\n",
    "sns.jointplot(x='question Tokens', y='answer tokens', data=NewData, kind='kde', fill=True, cmap='YlGnBu')\n",
    "\n",
    "plt.show()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def clean_text(text):\n",
    "    text=re.sub('-',' ',text.lower())\n",
    "    text=re.sub('[.]',' . ',text)\n",
    "    text=re.sub('[1]',' 1 ',text)\n",
    "    text=re.sub('[2]',' 2 ',text)\n",
    "    text=re.sub('[3]',' 3 ',text)\n",
    "    text=re.sub('[4]',' 4 ',text)\n",
    "    text=re.sub('[5]',' 5 ',text)\n",
    "    text=re.sub('[6]',' 6 ',text)\n",
    "    text=re.sub('[7]',' 7 ',text)\n",
    "    text=re.sub('[8]',' 8 ',text)\n",
    "    text=re.sub('[9]',' 9 ',text)\n",
    "    text=re.sub('[0]',' 0 ',text)\n",
    "    text=re.sub('[,]',' , ',text)\n",
    "    text=re.sub('[?]',' ? ',text)\n",
    "    text=re.sub('[!]',' ! ',text)\n",
    "    text=re.sub('[$]',' $ ',text)\n",
    "    text=re.sub('[&]',' & ',text)\n",
    "    text=re.sub('[/]',' / ',text)\n",
    "    text=re.sub('[:]',' : ',text)\n",
    "    text=re.sub('[;]',' ; ',text)\n",
    "    text=re.sub('[*]',' * ',text)\n",
    "    text=re.sub('[\\']',' \\' ',text)\n",
    "    text=re.sub('[\\\"]',' \\\" ',text)\n",
    "    text=re.sub('\\t',' ',text)\n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 268 entries, 0 to 267\n",
      "Data columns (total 4 columns):\n",
      " #   Column           Non-Null Count  Dtype \n",
      "---  ------           --------------  ----- \n",
      " 0   Patient          246 non-null    object\n",
      " 1   Doctor           258 non-null    object\n",
      " 2   question Tokens  268 non-null    int64 \n",
      " 3   answer tokens    268 non-null    int64 \n",
      "dtypes: int64(2), object(2)\n",
      "memory usage: 8.5+ KB\n"
     ]
    }
   ],
   "source": [
    "NewData.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Patient</th>\n",
       "      <th>Doctor</th>\n",
       "      <th>encoder_inputs</th>\n",
       "      <th>decoder_targets</th>\n",
       "      <th>decoder_inputs</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Doctor, I've been feeling really strange lately. I had a reaction after eating something, and my throat felt like it was closing up. I'm really worried. Can you help me?</td>\n",
       "      <td>Of course, I'll do my best to help you. It sounds like you may have experienced an allergic reaction. Can you tell me more about what happened?</td>\n",
       "      <td>doctor ,  i ' ve been feeling really strange lately .  i had a reaction after eating something ,  and my throat felt like it was closing up .  i ' m really worried .  can you help me ?</td>\n",
       "      <td>of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Well, I was at a restaurant and I had this dish with shrimp. Shortly after I finished eating, my lips started to swell, and I had difficulty breathing. I've never had this happen before, and it scared me.</td>\n",
       "      <td>I understand your concern. Based on your symptoms, it's possible that you had a severe allergic reaction called anaphylaxis. This is a serious condition that requires immediate medical attention. Have you experienced any other symptoms, such as hives, itching, or lightheadedness?</td>\n",
       "      <td>well ,  i was at a restaurant and i had this dish with shrimp .  shortly after i finished eating ,  my lips started to swell ,  and i had difficulty breathing .  i ' ve never had this happen before ,  and it scared me .</td>\n",
       "      <td>i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Yes, actually, I did notice some hives on my arms and chest, and I felt lightheaded and dizzy. I had no idea this could happen from eating shrimp. What exactly is anaphylaxis?</td>\n",
       "      <td>That's probably what's causing your reaction. Pork is a common allergen, and it can cause a variety of symptoms, including hives, difficulty breathing, and even anaphylaxis.</td>\n",
       "      <td>yes ,  actually ,  i did notice some hives on my arms and chest ,  and i felt lightheaded and dizzy .  i had no idea this could happen from eating shrimp .  what exactly is anaphylaxis ?</td>\n",
       "      <td>that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen, such as shrimp in your case. It can affect multiple systems in your body and can be life-threatening if not treated promptly. Common symptoms include swelling of the lips, face, or throat, difficulty breathing, hives or rash, and a drop in blood pressure leading to dizziness or fainting.</td>\n",
       "      <td>That sounds really serious! I had no idea an allergic reaction could be so dangerous. What should I do if it happens again?</td>\n",
       "      <td>anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen ,  such as shrimp in your case .  it can affect multiple systems in your body and can be life threatening if not treated promptly .  common symptoms include swelling of the lips ,  face ,  or throat ,  difficulty breathing ,  hives or rash ,  and a drop in blood pressure leading to dizziness or fainting .</td>\n",
       "      <td>that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Good morning, Doctor. I've been experiencing some unusual symptoms, and I'm not sure what's going on. I noticed some skin issues and recently had a blood test done. Can you please take a look at the reports and help me understand what's happening?</td>\n",
       "      <td>Good morning. Of course, I'll be happy to assist you. Please hand me your reports, and let's discuss your symptoms in detail. What specific skin issues have you been experiencing?</td>\n",
       "      <td>good morning ,  doctor .  i ' ve been experiencing some unusual symptoms ,  and i ' m not sure what ' s going on .  i noticed some skin issues and recently had a blood test done .  can you please take a look at the reports and help me understand what ' s happening ?</td>\n",
       "      <td>good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Well, I've been having frequent rashes and hives on different parts of my body. They appear as red, itchy patches, and they come and go randomly. It's quite uncomfortable, and I'm not sure what triggers them.</td>\n",
       "      <td>I see. Skin rashes and hives can be indicative of an allergic reaction. It's important to identify the underlying cause. Now, let's take a look at your blood test results. Could you please pass them to me?</td>\n",
       "      <td>well ,  i ' ve been having frequent rashes and hives on different parts of my body .  they appear as red ,  itchy patches ,  and they come and go randomly .  it ' s quite uncomfortable ,  and i ' m not sure what triggers them .</td>\n",
       "      <td>i see .  skin rashes and hives can be indicative of an allergic reaction .  it ' s important to identify the underlying cause .  now ,  let ' s take a look at your blood test results .  could you please pass them to me ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt;  i see .  skin rashes and hives can be indicative of an allergic reaction .  it ' s important to identify the underlying cause .  now ,  let ' s take a look at your blood test results .  could you please pass them to me ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Here are the reports, Doctor. I hope they can provide some insight into my condition.</td>\n",
       "      <td>Thank you. Let me review these reports.</td>\n",
       "      <td>here are the reports ,  doctor .  i hope they can provide some insight into my condition .</td>\n",
       "      <td>thank you .  let me review these reports .   &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; thank you .  let me review these reports .   &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Doctor, I had a really scary experience after eating pork yesterday. My face swelled up, and I had difficulty breathing. I think it might have been an allergic reaction. Can you help me understand what happened?</td>\n",
       "      <td>I'm sorry to hear about your distressing experience. Allergic reactions can indeed occur after consuming certain foods. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the facial swelling and difficulty breathing?</td>\n",
       "      <td>doctor ,  i had a really scary experience after eating pork yesterday .  my face swelled up ,  and i had difficulty breathing .  i think it might have been an allergic reaction .  can you help me understand what happened ?</td>\n",
       "      <td>i ' m sorry to hear about your distressing experience .  allergic reactions can indeed occur after consuming certain foods .  let ' s discuss your symptoms in more detail .  did you notice any other reactions apart from the facial swelling and difficulty breathing ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt;  i ' m sorry to hear about your distressing experience .  allergic reactions can indeed occur after consuming certain foods .  let ' s discuss your symptoms in more detail .  did you notice any other reactions apart from the facial swelling and difficulty breathing ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Yes, I also had hives on my body, and I felt quite dizzy. It was really frightening, and I had no idea that this could happen from eating pork. Is it possible that I have an allergy to pork?</td>\n",
       "      <td>It's possible that you have developed an allergy to pork. Allergic reactions can vary from person to person, and some individuals can be allergic to specific types of meat. To better understand your condition, it would be helpful to review your skin and blood test reports. Can you please provide me with those?</td>\n",
       "      <td>yes ,  i also had hives on my body ,  and i felt quite dizzy .  it was really frightening ,  and i had no idea that this could happen from eating pork .  is it possible that i have an allergy to pork ?</td>\n",
       "      <td>it ' s possible that you have developed an allergy to pork .  allergic reactions can vary from person to person ,  and some individuals can be allergic to specific types of meat .  to better understand your condition ,  it would be helpful to review your skin and blood test reports .  can you please provide me with those ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt;  it ' s possible that you have developed an allergy to pork .  allergic reactions can vary from person to person ,  and some individuals can be allergic to specific types of meat .  to better understand your condition ,  it would be helpful to review your skin and blood test reports .  can you please provide me with those ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Certainly, Doctor. Here are the reports. I hope they can shed some light on what's happening.</td>\n",
       "      <td>Thank you. Let me take a look at the report .Based on your blood test results, your IgE levels are elevated, indicating a possible allergic reaction. Additionally, your skin prick test shows a positive reaction to pork allergens, further suggesting an allergy to pork.</td>\n",
       "      <td>certainly ,  doctor .  here are the reports .  i hope they can shed some light on what ' s happening .</td>\n",
       "      <td>thank you .  let me take a look at the report  . based on your blood test results ,  your ige levels are elevated ,  indicating a possible allergic reaction .  additionally ,  your skin prick test shows a positive reaction to pork allergens ,  further suggesting an allergy to pork .  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; thank you .  let me take a look at the report  . based on your blood test results ,  your ige levels are elevated ,  indicating a possible allergic reaction .  additionally ,  your skin prick test shows a positive reaction to pork allergens ,  further suggesting an allergy to pork .  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                                                                                                                                                                                                                                                                                                                                                                   Patient  \\\n",
       "0                                                                                                                                                                                                                                                Doctor, I've been feeling really strange lately. I had a reaction after eating something, and my throat felt like it was closing up. I'm really worried. Can you help me?   \n",
       "1                                                                                                                                                                                                             Well, I was at a restaurant and I had this dish with shrimp. Shortly after I finished eating, my lips started to swell, and I had difficulty breathing. I've never had this happen before, and it scared me.   \n",
       "2                                                                                                                                                                                                                                          Yes, actually, I did notice some hives on my arms and chest, and I felt lightheaded and dizzy. I had no idea this could happen from eating shrimp. What exactly is anaphylaxis?   \n",
       "3  Anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen, such as shrimp in your case. It can affect multiple systems in your body and can be life-threatening if not treated promptly. Common symptoms include swelling of the lips, face, or throat, difficulty breathing, hives or rash, and a drop in blood pressure leading to dizziness or fainting.   \n",
       "4                                                                                                                                                                  Good morning, Doctor. I've been experiencing some unusual symptoms, and I'm not sure what's going on. I noticed some skin issues and recently had a blood test done. Can you please take a look at the reports and help me understand what's happening?   \n",
       "5                                                                                                                                                                                                         Well, I've been having frequent rashes and hives on different parts of my body. They appear as red, itchy patches, and they come and go randomly. It's quite uncomfortable, and I'm not sure what triggers them.   \n",
       "6                                                                                                                                                                                                                                                                                                                                    Here are the reports, Doctor. I hope they can provide some insight into my condition.   \n",
       "7                                                                                                                                                                                                      Doctor, I had a really scary experience after eating pork yesterday. My face swelled up, and I had difficulty breathing. I think it might have been an allergic reaction. Can you help me understand what happened?   \n",
       "8                                                                                                                                                                                                                           Yes, I also had hives on my body, and I felt quite dizzy. It was really frightening, and I had no idea that this could happen from eating pork. Is it possible that I have an allergy to pork?   \n",
       "9                                                                                                                                                                                                                                                                                                                            Certainly, Doctor. Here are the reports. I hope they can shed some light on what's happening.   \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                                     Doctor  \\\n",
       "0                                                                                                                                                                           Of course, I'll do my best to help you. It sounds like you may have experienced an allergic reaction. Can you tell me more about what happened?   \n",
       "1                                  I understand your concern. Based on your symptoms, it's possible that you had a severe allergic reaction called anaphylaxis. This is a serious condition that requires immediate medical attention. Have you experienced any other symptoms, such as hives, itching, or lightheadedness?   \n",
       "2                                                                                                                                             That's probably what's causing your reaction. Pork is a common allergen, and it can cause a variety of symptoms, including hives, difficulty breathing, and even anaphylaxis.   \n",
       "3                                                                                                                                                                                               That sounds really serious! I had no idea an allergic reaction could be so dangerous. What should I do if it happens again?   \n",
       "4                                                                                                                                       Good morning. Of course, I'll be happy to assist you. Please hand me your reports, and let's discuss your symptoms in detail. What specific skin issues have you been experiencing?   \n",
       "5                                                                                                             I see. Skin rashes and hives can be indicative of an allergic reaction. It's important to identify the underlying cause. Now, let's take a look at your blood test results. Could you please pass them to me?   \n",
       "6                                                                                                                                                                                                                                                                                  Thank you. Let me review these reports.    \n",
       "7                                                           I'm sorry to hear about your distressing experience. Allergic reactions can indeed occur after consuming certain foods. Let's discuss your symptoms in more detail. Did you notice any other reactions apart from the facial swelling and difficulty breathing?   \n",
       "8   It's possible that you have developed an allergy to pork. Allergic reactions can vary from person to person, and some individuals can be allergic to specific types of meat. To better understand your condition, it would be helpful to review your skin and blood test reports. Can you please provide me with those?   \n",
       "9                                              Thank you. Let me take a look at the report .Based on your blood test results, your IgE levels are elevated, indicating a possible allergic reaction. Additionally, your skin prick test shows a positive reaction to pork allergens, further suggesting an allergy to pork.   \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                                                                                                                                              encoder_inputs  \\\n",
       "0                                                                                                                                                                                                                                                  doctor ,  i ' ve been feeling really strange lately .  i had a reaction after eating something ,  and my throat felt like it was closing up .  i ' m really worried .  can you help me ?    \n",
       "1                                                                                                                                                                                                               well ,  i was at a restaurant and i had this dish with shrimp .  shortly after i finished eating ,  my lips started to swell ,  and i had difficulty breathing .  i ' ve never had this happen before ,  and it scared me .    \n",
       "2                                                                                                                                                                                                                                                yes ,  actually ,  i did notice some hives on my arms and chest ,  and i felt lightheaded and dizzy .  i had no idea this could happen from eating shrimp .  what exactly is anaphylaxis ?    \n",
       "3  anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen ,  such as shrimp in your case .  it can affect multiple systems in your body and can be life threatening if not treated promptly .  common symptoms include swelling of the lips ,  face ,  or throat ,  difficulty breathing ,  hives or rash ,  and a drop in blood pressure leading to dizziness or fainting .    \n",
       "4                                                                                                                                                                good morning ,  doctor .  i ' ve been experiencing some unusual symptoms ,  and i ' m not sure what ' s going on .  i noticed some skin issues and recently had a blood test done .  can you please take a look at the reports and help me understand what ' s happening ?    \n",
       "5                                                                                                                                                                                                       well ,  i ' ve been having frequent rashes and hives on different parts of my body .  they appear as red ,  itchy patches ,  and they come and go randomly .  it ' s quite uncomfortable ,  and i ' m not sure what triggers them .    \n",
       "6                                                                                                                                                                                                                                                                                                                                                here are the reports ,  doctor .  i hope they can provide some insight into my condition .    \n",
       "7                                                                                                                                                                                                            doctor ,  i had a really scary experience after eating pork yesterday .  my face swelled up ,  and i had difficulty breathing .  i think it might have been an allergic reaction .  can you help me understand what happened ?    \n",
       "8                                                                                                                                                                                                                                 yes ,  i also had hives on my body ,  and i felt quite dizzy .  it was really frightening ,  and i had no idea that this could happen from eating pork .  is it possible that i have an allergy to pork ?    \n",
       "9                                                                                                                                                                                                                                                                                                                                    certainly ,  doctor .  here are the reports .  i hope they can shed some light on what ' s happening .    \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                                                decoder_targets  \\\n",
       "0                                                                                                                                                                               of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  <end>   \n",
       "1                              i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  <end>   \n",
       "2                                                                                                                                           that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  <end>   \n",
       "3                                                                                                                                                                                                       that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  <end>   \n",
       "4                                                                                                                                     good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  <end>   \n",
       "5                                                                                                           i see .  skin rashes and hives can be indicative of an allergic reaction .  it ' s important to identify the underlying cause .  now ,  let ' s take a look at your blood test results .  could you please pass them to me ?  <end>   \n",
       "6                                                                                                                                                                                                                                                                                            thank you .  let me review these reports .   <end>   \n",
       "7                                                             i ' m sorry to hear about your distressing experience .  allergic reactions can indeed occur after consuming certain foods .  let ' s discuss your symptoms in more detail .  did you notice any other reactions apart from the facial swelling and difficulty breathing ?  <end>   \n",
       "8   it ' s possible that you have developed an allergy to pork .  allergic reactions can vary from person to person ,  and some individuals can be allergic to specific types of meat .  to better understand your condition ,  it would be helpful to review your skin and blood test reports .  can you please provide me with those ?  <end>   \n",
       "9                                            thank you .  let me take a look at the report  . based on your blood test results ,  your ige levels are elevated ,  indicating a possible allergic reaction .  additionally ,  your skin prick test shows a positive reaction to pork allergens ,  further suggesting an allergy to pork .  <end>   \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                                                         decoder_inputs  \n",
       "0                                                                                                                                                                               <start> of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  <end>  \n",
       "1                              <start> i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  <end>  \n",
       "2                                                                                                                                           <start> that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  <end>  \n",
       "3                                                                                                                                                                                                       <start> that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  <end>  \n",
       "4                                                                                                                                     <start> good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  <end>  \n",
       "5                                                                                                          <start>  i see .  skin rashes and hives can be indicative of an allergic reaction .  it ' s important to identify the underlying cause .  now ,  let ' s take a look at your blood test results .  could you please pass them to me ?  <end>  \n",
       "6                                                                                                                                                                                                                                                                                            <start> thank you .  let me review these reports .   <end>  \n",
       "7                                                            <start>  i ' m sorry to hear about your distressing experience .  allergic reactions can indeed occur after consuming certain foods .  let ' s discuss your symptoms in more detail .  did you notice any other reactions apart from the facial swelling and difficulty breathing ?  <end>  \n",
       "8  <start>  it ' s possible that you have developed an allergy to pork .  allergic reactions can vary from person to person ,  and some individuals can be allergic to specific types of meat .  to better understand your condition ,  it would be helpful to review your skin and blood test reports .  can you please provide me with those ?  <end>  \n",
       "9                                            <start> thank you .  let me take a look at the report  . based on your blood test results ,  your ige levels are elevated ,  indicating a possible allergic reaction .  additionally ,  your skin prick test shows a positive reaction to pork allergens ,  further suggesting an allergy to pork .  <end>  "
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NewData.drop(columns=['answer tokens', 'question Tokens'], inplace=True)\n",
    "NewData['encoder_inputs'] = NewData['Patient'].astype(str).apply(clean_text)\n",
    "NewData['decoder_targets'] = NewData['Doctor'].apply(lambda x: clean_text(x) if pd.notnull(x) else '')\n",
    "NewData['decoder_targets'] += ' <end>'\n",
    "NewData['decoder_inputs'] = '<start> ' + NewData['Doctor'].apply(lambda x: clean_text(x) if pd.notnull(x) else '')\n",
    "NewData['decoder_inputs'] += ' <end>'\n",
    "NewData.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABxoAAAHdCAYAAAAq63z1AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAAEAAElEQVR4nOzdeXxU9b3/8feZPZnJRkICIWwJIIKA4gK41N1WsdRWb221aq3We1FbFWu1Vlu73KK1tnhVulhtxaq1m9dauW0tuKAg7oAiAgn7Erask9nn/P7gBzXMOSF7Znk9Hw8etnPOzHzzneV85vv5fj9fo7Gx0RQAAAAAAAAAAAAAdIFjoBsAAAAAAAAAAAAAIPOQaAQAAAAAAAAAAADQZSQaAQAAAAAAAAAAAHQZiUYAAAAAAAAAAAAAXUaiEQAAAAAAAAAAAECXkWgEAAAAAAAAAAAA0GUkGgEAAAAAAAAAAAB0GYlGAAAAAAAAAAAAAF1GohEAAAAAAAAAAABAl5FoBAAAAAAAAAAAANBlJBoBAAAAAAAAAAAAdBmJxl4WDodVV1encDg80E1BH+O1zg28zrmD1zo38DrnDl5re/RNx+gfe/SNPfrGHn1jj75BruK9b4++sUff2KNv7NE39ugbe/RN15Fo7AOJRGKgm4B+wmudG3idcwevdW7gde4Z0zT117/+Veeff76OOOIIDR06VMcdd5xuvPFGbdy4MeX85uZm3X777TrqqKNUXl6uSZMm6c4771Rra2uft5XX2h590zH6xx59Y4++sUff2KNv0BOZFJcdive+PfrGHn1jj76xR9/Yo2/s0TddQ6IRAAAAnXbHHXfo8ssv1/r16zVz5kxdc801GjlypB577DGdcsopWr169cFzg8GgZs6cqfnz52vcuHG69tprNXbsWD3wwAOaNWsWswMBAAB6gLgMAACkA9dANwAAAACZob6+Xj//+c81fPhwvfrqqyoqKjp47KGHHtK3v/1tPfTQQ3rooYckSffff79WrVqlG2+8UXfdddfBc++66y7NmzdP8+fP15w5c/r7zwAAAMh4xGUAACBdsKIRAAAAnbJ582Ylk0lNnz693WCWJH3qU5+SJO3Zs0fS/lJejz/+uAKBgG655ZZ2595yyy0KBAJasGBB/zQcAAAgyxCXAQCAdEGiEQAAAJ1SU1Mjj8ej119/Xc3Nze2O/f3vf5cknXrqqZKk2tpa7dixQ9OmTZPf7293rt/v17Rp07Rx40Zt3bq1fxoPAACQRYjLAABAuqB0KgAAADpl0KBB+u53v6s77rhDJ5xwgs477zwVFBTo/fff1yuvvKKrr75a11xzjaT9A1qSVF1dbflY1dXVWrRokWpra1VVVdXh83Znz6BoNNruv/g3+qZj9I89+sYefWOPvrFH37Tn8/kGugkZJZPiskPx3rdH39ijb+zRN/boG3v0jT36putxGYlGAAAAdNp1112nyspKff3rX9ejjz568PYZM2booosuksu1P7w8MLP+0FJeBxQWFrY7ryPbt29XIpHoVnvr6+u7db9cQN90jP6xR9/Yo2/s0Tf26BvJ6XTaJsFgL9PiskPx3rdH39ijb+zRN/boG3v0jb1c7ZvuxGUkGgEAANBp99xzj37yk5/o9ttv1+c//3kVFRVp1apVuv3223X++edrwYIFOu+883r1OSsrK7t8n2g0qvr6elVUVMjj8fRqezIdfdMx+scefWOPvrFH39ijb9BTmRKXHYr3vj36xh59Y4++sUff2KNv7NE3XUeiEQAAAJ3y0ksvae7cubr22mt10003Hbx9xowZ+v3vf6+jjz5ad9xxh84777yDM+ObmposH+vAjPkD53WkJ6XUPB4Ppdhs0Dcdo3/s0Tf26Bt79I09+gbdkYlx2aF479ujb+zRN/boG3v0jT36xh5903mOgW4AAAAAMsMLL7wgSTrllFNSjlVUVGjs2LGqq6tTa2urampqJEl1dXWWj3Xg9gPnAQAAoPOIywAAQLog0QgAAIBOObAR+p49eyyP7927Vw6HQ263WzU1NRo6dKiWL1+uYDDY7rxgMKjly5dr5MiRqqqq6vN2AwAAZBviMgAAkC5INAIAAKBTpk+fLkmaP39+SumtRx99VNu2bdMJJ5wgr9crwzB02WWXqbW1Vffee2+7c++99161trbqiiuu6Le2AwAAZBPiMgAAkC7YoxEAAACdcsEFF+iRRx7R0qVLddxxx+ncc89VUVGRVqxYoVdeeUV5eXn67//+74Pn33DDDVq4cKHmzZunlStXasqUKVqxYoUWL16sqVOnavbs2QP41wAAAGQu4jIAAJAuSDQCAACgU5xOp5555hnNnz9fzzzzjP70pz8pGo2qvLxcn//853XzzTfriCOOOHi+3+/X888/r7vvvlvPPfeclixZooqKCl1//fW69dZblZeXN4B/DQAAQOYiLgMAAOmCRCMAAAA6zev16qabbtJNN93UqfOLioo0d+5czZ07t49bBgAAkFuIywAAQDpgj0YAAAAAAAAAAAAAXcaKxn7UGksqkjAHuhnd4nUaCrjJSwMAgOyRqbEZcRkAAMg2xGUAAGQuEo39KJIw9asPgwPdjG655ki/Au6BbgUAAEDvydTYjLgMAABkG+IyAAAyF1NuAAAAAAAAAAAAAHQZiUYAAAAAAAAAAAAAXUaiEQAAAAAAAAAAAECXkWgEAAAAAAAAAAAA0GUkGgEAAAAAAAAAAAB0GYlGAAAAAAAAAAAAAF1GohEAAAAAAAAAAABAl5FoBAAAAAAAAAAAANBlJBoBAAAAAAAAAAAAdBmJRgAAAAAAAAAAAABdRqIRAAAAAAAAAAAAQJeRaAQAAAAAAAAAAADQZSQaAQAAAAAAAAAAAHQZiUYAAAAAAAAAAAAAXUaiEQAAAAAAAAAAAECXkWgEAAAAAAAAAAAA0GUkGgEAAAAAAAAAAAB0GYlGAAAAAAAAAAAAAF1GohEAAAAAAAAAAABAl5FoBAAAAAAAAAAAANBlGZFonDRpkoqLiy3/zZw5M+X8SCSie+65R1OnTlVFRYXGjx+vG264Qbt37x6A1gMAAAAAAAAAAADZxzXQDeiswsJCzZ49O+X2ESNGtPv/yWRSl1xyiRYtWqTjjz9es2bNUm1trRYsWKCXX35Z//rXv1RWVtZfzQYAAAAAAAAAAACyUsYkGouKivStb33rsOc9+eSTWrRokS666CI9/PDDMgxDkvToo49qzpw5+uEPf6h58+b1cWsBAAAAAAAAAACA7JYRpVO7YsGCBZKk73znOweTjJJ05ZVXatSoUfrjH/+oUCg0UM0DAAAAAAAAAAAAskLGJBqj0aieeOIJ3XffffrVr36lt956K+WccDist956S2PHjk0pqWoYhk4//XQFg0G9++67/dVsAAAAAAAAAAAAICtlTOnU+vp6XXfdde1umzp1qh555BGNHj1akrRhwwYlk0lVV1dbPsaB22tra3XiiSd2+HzhcLhb7YxGo+3++3HJpEvxeLxbjzvQkkmz232SrTp6rZE9eJ1zB691bjj0dfb5fAPZHAAAAAAAACCjZUSi8dJLL9WMGTM0YcIE+f1+rV+/Xg899JCefvppzZo1S0uXLlVBQYGam5sl7d/P0UphYaEkHTyvI9u3b1cikeh2m+vr61Nuyy+rVHPT4Z87HUUjXm3Zs32gm5GWrF5rZB9e59zBa50b6uvr5XQ6bScnAQAAAAAAADi8jEg03nbbbe3+/+TJk/XLX/5SkvT000/rscce0/XXX9+rz1lZWdmt+0WjUdXX16uiokIej6fdsZakS4VFhb3RvH7n8XpVOnz4QDcjrXT0WiN78DrnDl7r3MDrDAAAAAAAAPSejEg02rnyyiv19NNPa/ny5br++usPrlhsamqyPP/ASsYD53Wkp6XUPB5PymMEwwm5XJnZ5Q6HQXk5G1avNbIPr3Pu4LXODbzOAAAAAAAAQM85BroBPVFaWipJamtrkySNGjVKDodDdXV1lucfuL2mpqZ/GggAAAAAAAAAAABkqYxONL711luSpBEjRkiS8vLydOyxx2rdunXavHlzu3NN09SLL74ov9+vY445pt/bCgAAAAAAAAAAAGSTtE80rl279uCKxUNvv+uuuyRJF1100cHbr7jiCknS97//fZmmefD23/zmN9q4caP+4z/+Q3l5eX3baAAAgCz0xBNPqLi4uMN/s2bNanef5uZm3X777TrqqKNUXl6uSZMm6c4771Rra+sA/RUAAADZgdgMAACkg7TfMPDPf/6z5s+frxNPPFHDhw9Xfn6+1q9frxdeeEGxWExz5szRSSeddPD8Sy65RM8884z+9Kc/adOmTTrppJNUV1en5557TiNHjtQdd9wxgH8NAABA5po0aZJuvfVWy2N//etf9eGHH+rMM888eFswGNTMmTO1atUqnXHGGbrooou0cuVKPfDAA3rttde0cOFC9soEAADoJmIzAACQDtI+0XjKKado7dq1WrlypZYtW6a2tjaVlpbq7LPP1tVXX60zzjij3fkOh0NPPvmkfvazn+npp5/W/PnzVVJSossuu0x33HGHysrKBugvAQAAyGyTJ0/W5MmTU26PRqN6+OGH5XK59MUvfvHg7ffff79WrVqlG2+88WAlCkm66667NG/ePM2fP19z5szpj6YDAABkHWIzAACQDtI+0XjyySfr5JNP7tJ9vF6vbrvtNt1222191CoAAAAc8Pzzz2vfvn2aOXOmysvLJe3fH/vxxx9XIBDQLbfc0u78W265Rb/+9a+1YMECBrMAAAB6GbEZ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61//up5//nmNHTtWK1eu1CuvvKLhw4frBz/4wcFzb7jhBi1cuFDz5s3TypUrNWXKFK1YsUKLFy/W1KlTNXv27AH8SwAAADIbcRkAAEgHGbeiMRaLafbs2Zo0aZJmzpxpec7999+vVatW6cYbb9Rf/vIX3XXXXfrLX/6iG2+8Ue+8847mz5/fz60GAADIDqNHj9aLL76oSy65RO+9955++ctfqq6uTl/96le1ePFiVVRUHDzX7/fr+eef1+zZs7V27Vo9+OCDWrt2ra6//no9++yzysvLG8C/BAAAILMRlwEAgHSQcSsaf/KTn2jNmjV6+eWXdf/996ccN01Tjz/+uAKBgG655ZZ2x2655Rb9+te/1oIFCzRnzpz+ajIAAEBWqaqq6vTEraKiIs2dO1dz587t41YBAADkHuIyAAAw0DJqReN7772n++67T7feeqvGjx9veU5tba127NihadOmtdvcWto/e2vatGnauHGjtm7d2h9NBgAAAAAAAAAAALJSxqxojEQiB0um3nDDDbbn1dbWSpKqq6stj1dXV2vRokWqra1VVVWV7eOEw+FutTMajbb778clky7F4/FuPe5ASybNbvdJturotUb24HXOHbzWueHQ15k9mwEAAAAAAIDuy5hE449+9CPV1tbqpZdektPptD2vublZ0v5yEFYKCwvbnWdn+/btSiQS3WytVF9fn3Jbflmlmps6ft50FY14tWXP9oFuRpcVllUoLvv3S884lV9WqZaEpFD33yt2XEqoeU/q+wgDw+ozjezEa50b6uvr5XQ6bScmAQAAAAAAADi8jEg0vvHGG3rggQd02223acKECf3ynJWVld26XzQaVX19vSoqKuTxeNoda0m6VFhU2BvN63cer1elw4cPdDO6rCXp0qMf9E1yN5FIKNgalD/g7zD53V3/NbFQwzOwz7NNR59pZBde69zA6wwAAAAAAAD0nrRPNMbjcc2ePVsTJ07UTTfddNjzD6xYbGpqsjx+YCXjgfPs9LSUmsfjSXmMYDghlyvtu9ySw2FkZHm5/uhzp9PZJ8+RqX2eraw+08hOvNa5gdcZAAAAAAAA6Lm0z3q1trYe3Hdx8ODBluecffbZkqTf/e53Gj9+vCSprq7O8twDt9fU1PR2UwEAAAAAAAAAAICckfaJRq/Xq8suu8zy2NKlS1VbW6tzzz1XZWVlGjFi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DgyAIDgOkiIiIiKiyjWljQZb43JxLs1g97jprylUfx4WDKWUeRnVXbWq0Gi9YzEzM1N0u/VORut+xanoVGoKhcLhOXR5Jshk7tnkAuCWsbsibqlUWiWv4a5tDrhv7MXFXVWfc2WpjW3uamFqoI9/HvalOsajNwObbukxtqknNPKCYqNEInDaTTcmdp0mIiKqSbwVEizp4YtRexynUP05MR/rruVgXFNOoUpERERU1eQSASt6+aHP1nsouizjxQwjFp7Nxlsdqn8gPVF1kZS8i/to3LgxAODGjRui262PW/cjIiIqrJGnBT1DxAufWQYL/ncjF/kmLvRNRERErtG/ngf+Ee0pum3miUzEZbvfGt1ERERE7qi1vxyvt/MS3fbxuWycTta7OCKimqPWFRrDwsJw4sQJ6HT266npdDqcOHECDRo0QERERDVFSERENV3nQBla+4kXG5NyzdgSlwuThcVGIiIico05nXwQ5un4011rtOCFX9JgMDMvISIiInKFaW290Eqkz8hkAV74JQ3ZBnM1REVU/WpVoVEQBIwdOxZarRYLFiyw27ZgwQJotVqMHz++mqIjIiJ3IAgCBtX3QIRafE3Om9km7I7Ph4XFRiIiInIBX6UEi7v7iW47nWzA/DPZLo6IiIiIqG5SSAumUBVbjjEu24TpMRkuj4moJqgZC2OVYO3atYiJiQEAXLx4EQCwbt06HDlyBADQrVs3jBs3DgAwZcoU7NixA4sXL8a5c+fQrl07nD17FgcOHECHDh0wYcKE6nkTRETkNmQSASMbqfD1tRyk5TuORjufZsDSCzrM7uRTDdERERFRXTOovgfGRnti3bUch20fn8tG33AleoQqqyEyIiIiorqlXYACb7T3wtzfHQd7bYjNxYMROXgsSnzqe6Layi3uaIyJicH69euxfv16nD17FgBw/Phx22PWIiQAqNVqbN++HRMmTMDVq1exbNkyXL16Fa+++iq2bNkClUpVXW+DiIjciEom4IkoFdQykWFqAJZc0GLtVZ3oNiIiIqLKNreLDxp7O864YLYALx9KR4bI4CgiIiIiqnyvtfVC12CF6LZpxzJwi+toUx3jFoXGlStXIiMjw+l/K1eutNvfx8cHc+fOxYULF5CcnIwLFy5gzpw58PISX6yViIhIjI9SgseiVJA7uVpOPZaBLXG5rg2KiIiI6iSNXILP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rl+/Lvr8Ze0D3blzJ44ePYrw8HAcOXIE586dw4EDB/D7778jISEB33//vcOgOyJn2NtKVIJr167hiy++gLe3N7777juHL9gnnngCM2fOhMViwZIlS0Sfw2Aw4JNPPrFLfgICAmy3+cfExNhdkPR6Pd5//30AwOuvv4433ngDnp6etu2CIKBPnz4YPXq07bFbt27ZinkLFy5Et27dbNu8vb3xySefoEGDBtBqtXadOBaLBYsWLQJQcEF78sknbdtUKhUWL16Mhg0bir6vjIwMLFiwAFKpFF9//TUGDx5st71///62zgrraxRlMpmwYMEC9OvXz/aYTCartDvOJBIJvvzyS0RFRdkea9iwId5++20AKDYBsVgs+OKLL+xGMrVo0cJ2F+mePXvcbgqBhQsXwmKxYODAgZg5cybkcrlt2+OPP27r8Fm8eLHotKZWRqMREyZMwEcffYSIiAjs2rXL4d/GrFmzYDQa8e6772LChAlQKBS2bU2aNMHatWuhVquxfv163LlzR/R1nnjiCUycONFuZP20adPQsmVLWCwW7N69227/a9eu2fZRqVR22+rXr48pU6aw0EhE5MasHVoA8Omnn9pd35s1a4YVK1bAYDCIHrthwwZcuHAB9913H7766iu70bxKpRKzZs3C4MGDkZqairVr19q2JScn23I860jvwtdPmUyGoUOH2s0YcezYMVtnyOrVq+1GlwcHB2Pt2rXw9vZGYmKi3WvpdDrbLBEzZ860y498fX3x+eefO71rrzJyVpPJhM8//xz33Xef7bGi19OK8Pb2xpo1axAaGmp7rF27drb8o7i8TKPR4PPPP4efn5/tsW7dumHevHkACjraCk875Q6sM348++yzeO6552w5ilQqxauvvorHH38cAGy/GZzRarUYPXo0NmzYgFatWmHv3r1o2rSpbbs1HwOAZcuW4amnnrLLhzp06IA1a9ZAEAQsX74cer3e4TUMBgOmTZtmiwkoOPfnzJmDwMBAaLVaHD582LYtJSUFaWlp8PHxwcsvv+yQ2zdt2hSvvPJKqdqJiIiqH3Mw987BFAoFnnnmGYSFhdk9LpVKMWLECEycOBEGgwH/+9//nD5H7969MWfOHNtdktYYs7Oz8fnnnwMoGFT2wAMP2Lb7+Phg9erVdrlfYZXRdlWpvOeTmFOnTmHQoEG4ffs2Jk+ejE8//dTu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",
      "text/plain": [
       "<Figure size 2000x500 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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xx83bhyio6Px9ttv4/PPP4dCoUC3bt0wY8YM3HLLLe7+eERERCQzCp1OZ5K6CPJtNTU1yM3NRXJyMu/O+C+2iXVsF0tsE0tsE5KCT89hIiIiIhIDAxMRERGRHQxMRERERHYwMBERERHZwcBEREREZAcDExEREZEdDExEREREdjAwEREREdnBwERERERkBwMTERERkR0MTERERER2MDARERER2cHARERERGQHAxMRERGRHQxMRERERHYwMBERERHZwcBEREREZAcDExEREZEdDExEREREdjAwEREREdnBwERERERkBwMTERERkR0MTERERER2MDARERER2cHARERERGQHAxMRERGRHQxMRERERHYwMBERERHZwcBEREREZAcDExEREZEdDExEREREdjAwEREREdnBwERERERkBwMTERERkR0MTERERER2MDARERER2cHARERERGQHAxMRERGRHQxMRERERHZ4ZWBatWoVnnnmGdxyyy2Ii4tDZGQkli9fbrFdXV0dVq9ejcceewy9e/dGq1atkJSUhGHDhuHf//436uvrLfa5cOECIiMjbf6ZM2eOJz4iERERyYhS6gKEmDVrFnJzcxEdHY34+Hjk5uZa3e7cuXOYPHkyNBoNBg8ejLvuugtlZWVYt24dnnvuOWzYsAErV66EQqGw2Ldz584YPny4xesDBw4U/fMQERGRvHllYJo/fz7S09ORkpKCd955B6+++qrV7TQaDd566y1MmDABarW64fVZs2ZhxIgRWL9+PVavXo1Ro0ZZ7NulSxe88MIL7voIRERE5EW8ckjulltuQUpKit3tEhMTMWXKlCZhCQDUajWmTZsGANixY4dbaiQiIiLf4bYeJp1Oh/z8fLRp0wbBwcHuOo1gQUFBAIDAwECr71+5cgWLFy9GWVkZYmNjMWjQIKSlpXmyRCIiIpIJwYHp8OHD+P7779GvXz8MHTq04fXq6mo8+eST+PbbbwEAWq0W8+bNw7333ut6tSL67LPPAKBJ7Y1t2bIFW7Zsafi7QqHAmDFj8M4771j0WNlSU1PjeqE+QK/XN/kvsU1sYbtYYptY8vc2CQkJcWl/XpuacrQ9BQemzz77DP/+97/x5ZdfNnl99uzZ+Oabbxr+rtPpMHXqVGRkZCAzM1Po6US1dOlSbNy4EYMHD8Ydd9zR5L2wsDDMmDEDw4cPR1paGkwmEw4fPozXXnsNX3zxBaqrq/Hpp586dJ78/Hyrd+L5q4KCAqlLkB22iXVsF0tsE0v+2iYZGRku7c9rU1OOtqfgwLRz506EhITg1ltvbXhNr9fjk08+QVBQEJYvX47evXtjzpw5+Oijj/Dhhx/ivffeE3o60axbtw4zZsxAcnIyFi1aZPF+bGwsXnrppSavDRkyBL169cKQIUOwdu1aHDp0CN27d7d7rsTERLHK9mp6vR4FBQWIj4+HSqWSuhxZYJtYx3axxDaxxDZxDa9NwggOTIWFhWjZsiUCAv43b3zv3r0oLy/HyJEjcfvttwMAXn75ZXz22WeymFy9YcMGTJ48GXFxcVi7di0SEhIc3jcsLAzjxo3DrFmzsGfPHocCk6vdpr5GpVKxTW7ANrGO7WKJbWKJbSIM20wYwXfJ6XQ6REVFNXlt7969UCgUGDZsWMNroaGhSE1NRX5+vvAqRbB+/XpMmjQJ0dHRWLt2LVJTU50+RnR0NACgqqpK5OqIiIhIzgQHptDQUBQVFTV5bdeuXQCAPn36NHldpVI16YnytPXr1yMrKwtRUVFYu3Yt0tPTBR1n//79AODQkgZERETkOwSnmHbt2uHixYs4fvw4AKC4uBi//voroqOj0b59+ybbXr58GTExMa5VKtDGjRuRlZWFyMhIrF27Fm3atGl2+8OHD8NkMlm8vmbNGqxYsQKRkZG47bbb3FUuERERyZDgOUyjRo3CgQMHMGbMGNx7773YsmUL9Ho9Ro8e3WS73NxcXLlyBbfccourtTZYtmxZQ29WdnY2AODTTz/F9u3bAQD9+vVDVlYWTp06hYceegi1tbUYOHAgvvrqK4tjpaSkYOLEiQ1/f/HFF3H+/Hn06tULiYmJqK+vx5EjR7Br1y4EBwdj4cKF0Gq1on0WIiIikj/BgemRRx7BTz/9hJ07d2LhwoUArt+aN3PmzCbbmddjGjRokAtlNrVr1y6sWLGiyWu7d+/G7t27G/6elZWFgoIC1NbWAgC+/vprq8caMGBAk8A0btw4rFmzBvv370dxcTGMRiNatmyJrKwsPPnkk2jXrp1on4OIiIi8g0Kn01mOPznIaDTip59+wunTp5GcnIzhw4dbzL5fsGAB8vLyMGXKFLvDYeSbampqkJubi+TkZN6d8V9sE+vYLpbYJpbYJiQFlx6NEhAQgOHDhze7jfmZbURERETeyisfvktERETkSaI9fFen06GiosLqHWZmycnJYp2OiIiIyGNcCkx5eXmYPXs21q1bB51O1+y2CoUCxcXFrpyOiIiISBKCA1NOTg7uuOMOlJSUNNurZObINkRERERyJDgwzZo1C8XFxcjIyMDf//539O7dG3FxcVAoFGLWR0RERCQ5wYFp27ZtCAoKwldffcVHhRAREZFPE3yXXEVFBdq2bcuwRERERD5PcGBKTk7mvCQiIiLyC4ID03333YdTp07h/PnzIpZDREREJD+CA9P06dORmZmJP/3pT7hw4YKYNRERERHJiuBJ3++++y4GDx6MxYsXo2/fvhg6dCjatm2LsLAwm/vc+GBeIiIiIm8gODC9/vrrUCgUMJlMqKurw48//mhzSQGTyQSFQsHARERERF5JcGAaP34811wiIiIivyA4MH3wwQdi1kFEREQkW4InfRMRERH5CwYmIiIiIjtcDkw5OTmYMWMGevfujVatWiE6OrrJ+8uWLcPcuXNRUVHh6qmIiIiIJCF4DhMAfPvtt5g2bRpqamoaVv2+cSK4TqfD3Llz0b59e4waNcqV0xEREZGLDEYjlAEcYHKW4BY7duwYHn30UdTW1mLq1Kn4/vvv0b17d4vtRo4cCZPJhB9//NGVOomIiEgEVfV6qUvwSoJ7mN577z0YDAbMnj0bjz32GAAgJCTEYrvU1FTExMTgwIEDwqskIiIiUVTU6RERZHm9puYJ7mHavn07NBpNQ1hqTqtWrXDlyhWhpyIiIiKRVNTXSF2CVxIcmIqKipCenu7QtoGBgTAYDEJPRURERCIpq+OQnBCCA1N4eDiuXr3q0La5ubkWd88RERGR55XWVUtdglcSHJg6deqEy5cv4+TJk81ut3v3bly9ehU9e/YUeioiIiISSWkdh+SEEByYxo4dC5PJhOnTp6O8vNzqNkVFRXjmmWegUCgwduxYwUUSERGROCoNHJITQvBdcg8++CCWL1+OnTt3YuDAgbj//vsbhug+//xz/P7771i5ciVKSkpw6623YuTIkaIVTURERMIY/7tuIjlHcGAKCAjAihUrMHXqVPz888+YN29ew3tPPvkkAMBkMmHo0KFYsmSJy4USEQkRrDnu1Pa1FR3dVAmRfNSbjAhUcPFKZ7i00ndkZCS+/PJLbN26Fd988w2OHTsGnU4HtVqNzMxM3HfffbjzzjvFqpWIyCZng5Ejx2F4Il8UplShtK4GLVRhUpfiVQQHpqKiIsTExAAAhgwZgiFDhjS7/bZt2zB48GChpyMiaiBWOHL0PAxO5Es0ShWKaisZmJwkuD9uzJgxqKqqcmjbHTt24MEHHxR6KiLyc8Ga403+SHF+Il8RrgxGQY31m7XINsGB6dChQ8jKykJ9fX2z2+3duxfjx493OFwREWljzkkakKyRSx1ErtIGheJKTYXUZXgdwYFp8ODB2Lx5M6ZNm2Zzm/379+OBBx5ARUUF/vKXvwg9FRH5OHMw0sacQ+ce8n0qAEMT+QK1Mgg6Ll7pNMGB6bPPPkOnTp3wxRdf4JVXXrF4/7fffsP999+P8vJyPPPMM3jxxRddqZOIfIzcepAc5W31Et1IAQXqTUaYuLyAU1x6NMpXX32FpKQkvPfee/joo48a3jty5AhGjx6NsrIyPPHEE3j55ZdFKZaIvJfU85DE5O31E4UGBqHcUCt1GV7FpWUF4uPj8fXXX+POO+/Eiy++iISEBGRkZOC+++6DTqfDlClT8M9//lOsWonIizBUEMlXlCoU+dVliAgKkboUr+FSYAKAjIwMrFq1CqNGjcKjjz4KjUaDkpISZGVl4c033xSjRiLyEgxJRN4hWqVGbpUOHSLipC7Fa4iyzGevXr2wZMkSGAwGlJSU4MEHH8S7774rxqGJSOZ8ZZjNWf72ecm3RKvCkFddKnUZXsWhHqa5c+c6dLBOnTrhwoULSEpKsrrPzJkznauOiGSHQYHI+wUFBKLWKN87UuXIocD0+uuvQ6FQOHRAk8lkMRRnMpmgUChEC0yrVq3Crl27cOjQIWRnZ0Ov12PBggWYOHGi1e3Lysrw+uuvY82aNSgsLER8fDxGjRqFmTNnQqPRWGxvNBqxePFifPLJJ8jJyYFarcYtt9yCv//970hNTRXlMxB5E4YkIt8TGhiE0roaaDmPySEOBabx48c7HJg8YdasWcjNzUV0dDTi4+ORm5trc9vKykoMHz4cR48exdChQ/HAAw/gyJEjmD9/Pnbs2IEff/wRISFNv1ieeeYZLFu2DB07dsSjjz6Ky5cv47vvvsPmzZvx888/o02bNu7+iESSY0gi8m2xwRqcryxBt8hEqUvxCg4Fpg8++MDddThl/vz5SE9PR0pKCt555x28+uqrNrd99913cfToUTzzzDNN1ot65ZVXMG/ePCxcuBDTp09veH3btm1YtmwZ+vfvj++++w4qlQrA9UfBjBkzBjNmzMA333zjts9GJCWGJCL/ER+swdmKYgYmB4ky6dvTbrnlFqSkpNjdzmQy4dNPP4VGo8GMGTOavDdjxgxoNBosW7asyevmv7/00ksNYQkAbr/9dgwcOBCbN29utkeLyNv466RtIn+nDQpBYS0fkeIorwxMjjp79iwuX76MPn36QK1WN3lPrVajT58+OH/+PPLy8hpe3759O9RqNfr27WtxvGHDhgG4/jBhIm/GkERECoUCSkUAqgx1UpfiFVxeh6m8vBzLli3Dhg0bcPr0aVRUVECj0aBdu3a488478dBDDyE8PFyMWp129uxZAEB6errV99PT07Fp0yacPXsWSUlJqKysxJUrV5CZmYnAwECr2zc+rj01NTUCK/cter2+yX9JmjbRxpzz2Ln8hSe+x/n9Y8nf2+TGebfOqqurQ73JCACICQrDiWuXkRnuv+sxOdqeLgWmgwcPIisrC/n5+U2eSVNeXo7Lly9j27ZtWLBgAT799FP06NHDlVMJUlZWBgDQarVW34+IiGiynfm/5tftbW9Pfn4+6uvrHS/YxxUUFEhdguy4s03k/BBbX+HJ4Xl+/1jy1zbJyMhwaf+ioquo+++1SWUy4LeyCwjX+e9jUhxtT8GBqaCgAGPGjEFJSQnCw8MxadIkZGZmIiEhAVeuXEF2djY+++wzXLp0CWPGjMGOHTsQHx8v9HReKTGRE+mA678FFhQUID4+vsm8MH/mjjZhD5LnJScnu/0c/P6xxDZxTUxMbEMPk8lkwrmScx75WvZ2ggPTe++9h5KSEgwZMgRLly5FZGSkxTbPP/88/vjHP2Lr1q2YP38+Zs2a5UqtTjP3CJWWWl/N9MYeJXs9SPZ6oG7karepr1GpVGyTG7jSJpx/JD1Pfj3z+8cS20SYoKAgBOJ/o0IRqlCUK+oRG6xuZi8SPOl748aNUKlU+Pjjj62GJeD6UNiiRYugVCqxYcMGoacSzLxeUk5OjtX3za+bt1Or1UhISMCFCxesDqXduD2RpzSepM3J2kQkpqTQCPxeekXqMmRPcGDKy8tDx44dERMT0+x2sbGx6NixY5M70TylTZs2aNmyJfbs2YPKysom71VWVmLPnj1o3bo1kpKSGl4fMGAAKisrsXv3bovjbdq0CQDQv39/9xZOfo3hiIg8KTFEi9MVRVKXIXuCA5NSqURtrWOTxPR6PZRKl2/Ic5pCocCkSZNQUVFh8biWN998ExUVFZg8eXKT181//+c//9nkDoyNGzdi+/btGDp0qENrQBHZ07mHAdqYcwxHRCQpZUAAFAAqDf5516GjBKeYNm3a4MiRIzh58iTat29vc7sTJ07g5MmT6Natm9BTWVi2bBl27doFAMjOzgYAfPrpp9i+fTsAoF+/fsjKygIAPP300/jxxx8xb948HDlyBN26dcPhw4exefNm9OzZE48//niTYw8ePBhZWVlYtmwZhgwZgjvuuANXrlzBt99+i6ioKLzxxhuifQ7yHzcGoWDLRxgSEUkmOSwSx0ovo090a6lLkS3BgWnkyJE4dOgQJk2ahEWLFqF79+4W2xw6dAhTpkwBANx7772Ci7zRrl27sGLFiiav7d69u8kwmjkwqdVq/PDDD3j99dexdu1a/Prrr4iPj8eTTz6JmTNnIjQ01OL48+bNQ2ZmJj755BN8+OGHUKvVGDFiBP7+978jLS1NtM9Bvom9RP6htqKj1CUQiSYlLAo7is4zMDVDodPpTPY3s1RVVYWhQ4fi5MmTUCgU6Nu3LzIzMxEXF4fCwkJkZ2dj9+7dMJlM6NixIzZt2mQ1nJDvq6mpQW5uLpKTk33yjhYGJP/kqcDk698/QrBNXHOuogRGWF76txSewfiU7tAogyWoSv4E9zCFhYXhu+++w5QpU7Bjxw7s2rWrSQ+PeSHLgQMHYvHixQxL5DMYkIjIF6WqW+DgtUsYHGv96Rj+zqWZ2AkJCfj++++xa9cuq49GueOOO6w+k43I2zAkieNK1VWrryeExXq4EiK6UUpYJDYXnmVgskGUW9f69euHfv36iXEoItlgSHKerUAkdD8GKSLPCVQEIFypwuXqMrQMdWyBZn8ieFmBuXPnYvny5Q5tu2LFCsydO1foqYg8hrf3O+dK1dUmf9x1fCLyjIzwWOwsviB1GbIkODC9/vrr+Oyzzxzadvny5QxMJGsMSY5xd0Cyd14icq9oVRgKayqgN/Lh3TfyyGqS5gngRHLCgGSf3ELKlaqrHKYjcrM0dQscuJaHftGpUpciKx4JTEVFRQgLC/PEqYjsYlCyTW4ByRqGJiL3Ste0wM8Fp9G3RWsoFAqpy5ENhwNTWVkZSktLm7ym1+uRm5trc5/q6mr88ssvOHXqFLp06SK8SiIRMChZ8oaAZA1DE5H7BCoCEB+iwcnyq+gQESd1ObLhcGBauHChxWNBfvvtN4cfeTJu3DjnKiMSCYPS/3hrQLKGoYnIfTqEx+HXonMMTI04HJhMJlOTuUgKhcLu3KSwsDCkpaVh/PjxeOKJJ4RXSSSAvwclXwpHRORZIYFBUAeqcL6iBKmaFlKXIwuCH40SFRWFvn374qeffhK7JvIxnn6Mgb8EJQai66TqZeKjUaTDNnGNrUej3KjSoMf+a7n4U1pvD1Qlf4Infc+cORNJSUli1kLkMm8OSwxAwkg1NBesOc4H8JJPUytVCA5QspfpvwQHpr/+9a9i1kHkEm8ISgxERORtumpb4ufC0/izurff3zHnkWUFiNxFjkGJwch/sJeJfF2YUoWIoBCcKL+Kjn4+AZyBibyWXMISA5L0eMcckft00SZgS+EZtAuPQaBC8ANCvJ7/fnLyWnJ4jIkUjwcheZL6a5HI3VQBSrRWR2GXnz9jjoGJvIqUFyeGJHmT8t+FoYl8XTtNLA7r8lFp0EtdimQYmMhrSHFRYkgiRzE0kS8LUCjQI7IV1uZnS12KZBiYSPakGIJjSPJOUv+byWG4mMhd4kPCUWeqx9nyIqlLkQQDE8maJy8+7E0isTA4ka+6KSoJ6wpOQm+sl7oUjxMcmFq0aIG7777boW1HjBiB6OhooaciP+WpCw5Dkm+R07+lOTgxPJGvCA5QolNEPH667H9f04KXFbjx2XKObE/kKE+GJSJPsPY1zTWcyBslh0XhQtE5nC4vQkZ4jNTleIxH1mGqra2FUskln8gxnghLDEq+zVvWZbL3tc5ARXLVu0UK1hecRFKYFqGBQVKX4xFuTzGVlZU4ffo0h+TIIe4OS54MSidKxfv26qA1iHYsoYR8HjnU7c2uD+cB2hgAOGd1G4YqkoIqIBA9I1vhq7wjeCilp188NsXhn4A//PADfvzxxyav5eTkYNq0aTb3qa6uxv79+1FWVoZbbrlFcJHkH7w5LIkZjpw9vpihROzPcePxPBmgvKWXyVW2vm8YpMjd4kPCUVBTjh3F5zEwJk3qctzO4Z+OR48exeeff97ktcLCQovXrImLi8MLL7zgfHXkN9wZltwVlNwdkhwllzocYa7VU8HJX0KTNTd+TzFAkTt00bbElqtnkRIaiRR1lNTluJXDP2kHDhzY5O9z585FUlISJk6caHV7hUKBsLAwpKWlYejQoQgLC3OtUvJZ3hSWvCmcyJkng5M/h6bGGn+fMTyRWBQKBQZEp2LN5Wz8MfVmaJTBUpfkNgqdTifo9rWoqCj07dsXP/30k9g1kY+pqalBbm4ukpOTERIS0uQ9bwhLDEnu5aneJoYmS94anJr7mUL2nasogRHi3rleXFuJw6WX8ae0Xj77gF7BV4Jr166JWQf5IbmHJQYlz/BUbxN7miyZvwe9NTiRfEQHq5GmboHVl37H6KQuUpfjFqLFQJPJhOLiYuTm5op1SPJh7gpLYixCeaJUybAkAU+0OZeTsI6La5IY0tQtUG8y4der1u/o9HYuB6adO3di3LhxSEpKQkZGBrp3797k/Xnz5mHatGnskaIG7gxLrmJQkhZDk7QYmshVPSITcbqiCMfLCqQuRXQu/XSaP38+XnnlFRiNRpvbaDQarFixAgMGDMCDDz7oyunIB8g1LLnjQn2o2PYxu0dzfSJbTpQqPTI8B3BekzUcpiNXmCeBb756BuHKYCSFRUpdkmgE9zDt3LkTL7/8MkJCQjBr1iwcOXIEffr0sdhuxIgRMJlMnBxObiN1WDpUrLT6R+x9/ImnhkX5HEHb2NtEQikDAjA4Jh3f5f+OEn2V1OWIRvBPpAULFgAA3n33XTzwwAMAYHWlz4SEBLRs2RJHjhwReiryEdoY8ce1XbnYuXJBdle4aXxc9kJ5prcJYI+TLcGa4+xpIkFCApUYGJOGFRd/w2QfWW5AcA/Tvn37EBUV1RCWmpOQkIDCwkKhpyIf0LmH+Bc9T4clT/cEsefpOk/OKzP3OLHX6X/Y00RChSuD0btFCj69cBA19XVSl+MywYFJp9MhOTlZzFqIPMLZC7AcQoscapCSFJPxG4cnBigiYVqowtBN2xLLLhxAnbFe6nJcIjgwRUZGIj8/36Ftz507h9hYdnX7KzkNxQkJS3Liz8FJ6uUe/DlAsZeJXBEfEo724XH47MJBGJq5SUzuBAemHj16oKioCPv27Wt2u/Xr10On06F3795CT0VezB0/aD0RluQeTPx5uE4uSz/cGKB8PUwxNJErkkK1aB0Whc8v/oZ6k3eGJsE/eSZOnIgNGzbgmWeewRdffIFWrVpZbHPq1ClMnz4dCoUCkyZNcqlQVyxfvhzTpk1rdpvBgwdjzZo1AIA5c+Zg7ty5Nrc9fPgwWrduLWqN5Bh3hyUhAeRMeZDT+7QNF28831yzuyaJC2kTd09Y9/RDfJ1l6+uUk8rJn7VWR6EeRqzKPYzxyd0RYOVGMTkTHJhGjhyJe+65B2vXrkX//v1x++23Iy8vDwDw+uuv4/fff8eGDRug1+sxduxYDBkyRLSindWlSxfMnDnT6ntr1qzB8ePHMWzYMIv3JkyYgJSUFIvXtVqt6DX6Irn8RuqOsCQkJNnaX6zwZK1+R4KLO3qpbjymuwKUp+6iE4u1IOVNIYp3zZGr0tXRMJpM+DL3MMYkd/Oq0CT44bsAoNfr8cILL2Dp0qVNFq9UKBQwmUxQKBTIysrCm2++iaAg1y4w7qDX69GhQweUlZUhOzsbcXFxAP7Xw7R27VoMGjRI4iq9l9iBSUjvkthhydWgZI+YPU9y5a7w5E3BqTlyD1ByCEx8+K5r3PHwXWedKr+KCoMe9yd18ZrQ5NKvliqVCm+//TYef/xxrF69GseOHYNOp4NarUZmZiZGjRqFzMxMsWoV3Q8//ICSkhIMHz68ISyROHwtLLk7KDU+j6+HJncNIcp9mM5Rjb/W5Rie2MtEYmgXHouT5YX4Ju8o7k/qYnUdR7kRpS++bdu2eO6558Q4lEctW7YMAJCVlWX1/Z07d+LAgQMICAhAeno6brnlFmg0Gk+WSAKJFZY8FZSsnZPBSZjG//YMT0Ty1T48DifKCvHNpaMY3Ur+oUket5tI4OLFi9i6dStatWqF2267zeo2c+bMafJ3rVaL119/HRMmTPBEiV5L6t4lbw5L1s7P4CScL4YnBifyJR0i4nC8rADfXjqG+1p1lnVo8tvAtHz5chiNRkyYMAGBgYFN3uvcuTPef/99DBw4EAkJCSgoKMD69esxe/ZsPPHEE9Bqtbj77rvtnqOmpsZd5ctasIidcO66Rbu5sCR1ULqRPwzTAe6/289XwpMcglOw5jhKi9IkO79er2/yX3/j6ryturo6Wd3a3yYkCicqruKri4cxIq69x0OTo+0peNJ3t27dHN42MDAQ4eHhaN26Nfr374/x48cjMjJSyGlFYTQa0bVrV1y6dAm//fYbUlNTHdpv69atGDVqFDp27IidO3fa3T4nJwf19d69sqmzxH4Eijt6l7wpLN3IH4KTmaefpeeNIUrK0HTsN7/9fVtyGRkZLu2/8/Qx1Mnw2nShvgL1MGFAUJxHQ5Oj7Sk4MEVFRQnZDQqFApGRkfjggw9w5513CjqGqzZv3ozRo0djyJAhWL16tVP79uzZEzk5Obh48SIiIiKa3dYfe5jEXNXbG8LS2TLH92kTIV7Y8VRwEhogxa5PqgcRe0OIkio0Sd3DVFBQgPj4eKhUKsnqkIqrPUynrhXIqoepsZMVV2EwGTHcgz1Njran4F8R1q5di4MHD2LWrFlISEjA+PHj0bVrV4SHh6O8vBxHjx7FqlWrcPnyZbz00kto3749Tp06hRUrVuD48eP44x//iG3btrmclIWwN9m7OdHR0cjJyUF1dbXdwMTbXb2HM8HAmZBkbT8xgpPY6ziJ3bN24/FcrdHdw3W22ArhcgpSV6quShKatDHnJL9bTqVS8eesAEFBQQiUeFkBW7q0aIXfy65gfclZ3JvYSVZzmgT3MJ04cQK33XYbhg0bhkWLFiE4ONhiG71ej0ceeQQbN27Ezz//jI4dO8JgMODPf/4z1qxZg8mTJ2PevHmufganlJSUoEOHDlCr1Thx4oTVum2prKxEhw4dYDQaceHCBSiV7JJuTMrJ3p7qXRIalqwRs8fpRjcGFDkNNYoR8KTqcWqO1CFKitAkVWDiOkyukcM6TPYcK70CABiZmCmb0CT4WXJz586FyWTC+++/bzN0qFQqvPfeew3bA4BSqcRbb72FwMBAbNu2TejpBVu5cmXD6uPW6i4vL8eZM2csXq+ursbTTz+N8vJyjBo1imHJzcSe7O1qWDpbFiRqWDIf013OlAc1+SMnYtQlx2fpmR8OLNWz7nz1GXbknzprE2CCCT9cPg6TSR7hTvB39o4dO9C+fXuEh4c3u11ERATat2/fZJJ0bGwsMjIycOHCBaGnF+yzzz4DYHs4rqSkBL169ULPnj3Rrl07xMfHo7CwEFu3bsWlS5eQmZmJ1157zZMlkx2uXKAcDUvuIuYwnTcSY2jRU49hcYZUd+RJNTxH5A5dtC1xWJePn66cxN0tO0hdjvDAVFZWhtLSUoe2LS0tRXl5eZPXNBqNx7vZDhw4gOzsbNx0003o1KmT1W2ioqIwZcoUHDhwABs3boROp0NoaCjatWuHRx99FFOnTkVoaKhH6/YGYg7Heap3SeqwdON5/DU0mYm17pTcApSvLGdAJIVukYk4eO0SNl45hdsT2klai+DAlJqailOnTmHHjh0YMGCAze127NiBnJwctGvX9INevnxZ8J12Qt10003Q6XTNbhMREYE333zTMwWRy9w5/OGpsHTj+RicxJ3QbiswSxGkPPH4FvYyka/pEZmI/dfy8EvhWdwS10ayOgTPYRo7dixMJhMeeughfPXVVxbrDdXX1+Prr7/GpEmToFAoMH78+Ib3zp492zC8ReROQnuXPB2W5HLu5pjncd34x53cOQ+r8TwoT8+HkmqeE5E3UigUuDkqCZeqS7G72PNTecwEf9c++eST+Pnnn7Fr1y488sgjmD59Ojp06ACNRoOKigqcOHECFRUVMJlMGDBgAJ588smGfVetWoUWLVo4tFo2+Rc5TFyVQ2CRqrdJyGdvbh8x6xe758kaa6HJ3SuPu6O3ib1M5GsUCgX6tEjB9qLzCAlUontkK8/XIHRZAeD6rZ2zZ8/GkiVLUFlZafG+Wq3GlClT8Ne//pW3fvowqeYv2fst3Rt7l2xxR3CS4nN6cikFd3FHgHJHaPJEYOKyAt7JG5YVsMVoMuGXq2cxOCYN7SPiPHpulwKTWUVFBXbv3o3Tp0+jsrISarUaGRkZ6NevH9RqtRh1koz5UmCSY1i6kbOhQ66fyRO9Z+4OUWKGJ7FDEwMT2eLNgQkADEYjNl89gxEJHZCs9txcaMGBae7cuVAoFHj66aedWvyRfI9YgUnsR6FYC0ze2LvkLzw5/CjXR7eIGZoYmMgWbw9MAFBbb8Dmq2cwIbk7ooM90zEjODBFR0ejTZs22Lt3r9g1kZeRIjD5a++Sv5DqTkFXg5QYwUms0MTARLb4QmACgApDLbYXncMfU3tBrXT/MwUFT/qOiYmBRqMRsxYir1RcbP0hltHRgm9C9Xs3hldPBShXn4En1TPvrOHEb/J1GmUwekUlY/nFg/hTai8oAwLdej7BP9H79OmDM2fOQK/Xi1kPeRmxnx/nTmL3LhUXG22GJUfeJ8d5ejkDM6FLGriyRAGXHCByXHSwGhmaGHyZd8Ttj1AR/J359NNP46effsI///lPvPrqq2LWRH7IE8NxYnImCJm39YYeJzEDnrs/r6eWMwCELWlwqFgpi54mIl+XEhaF0roa/HL1LG6Na+u28wi+ssTFxeHll1/Gq6++iuzsbDz00EPo0KEDwsLCbO6TnJws9HREsiE0VMgxOLmzB+zGY3vyc9sKU2IEKWce4cLQROQZnSMSsL3oHE6UFaKDm5YbEByYunXr1vD/mzZtwqZNm5rdXqFQoLi4WOjpiNzG05O9pQxOUg4RWju3p9vA2r+10BB1pjzIbaHpRKmSz50jcoJCoUC/6FT8XHga8SHhiFKJ/8xXwT+tTCaTU3+MRs7lIOmI9WgNMQOHJ+Y4mc8h1/lUcqjPlXlRjn5defKxK0T+ShkQgP7RrfFF7iEY3JA5BH8XX7t2Tcw6iEThjRcmMXuc5BiKnNFc/Z7ojWocmhzteXK0p8nT3H2HnFRLChA1JyIoBG00Mfjpygnckyju82q97+pCPkfMCd/u5IneoBvZCgneHoyEsPeZxQ5UzjzPz5HQxPlMRJ6Rpm6BHUXncKr8KtqFi/eLAwMTCeZNSwp4K38MRkK5az2ss2VBooUmIvKM3i1SsKHgFFLCIhESKM6UDAYmIrJQWuDahV8bL59V08WYbM7QRORdggIC0U2biDX52Rib3M3+Dg5wOTDl5+fjq6++wpEjR1BSUoK6Ous/LBQKBdasWePq6YhExcehXOdqQHLmeHIIU0LmjTkamohIHhJDI3C+sgQ5FcVI10S7fDyXAtPSpUsxc+bMJiGp8UqbCoWi4TXz/xN5mhh3yPni0JjYIUms83oyUDX+d3UkPLkamjiPicizbopKwrorJ/Fom74IVLg2PC84MO3atQvTp09HaGgonnrqKXz33XfIycnB/Pnzce3aNezbtw/r1q2DUqnE888/j7g49ywkRUSOkyokOcPZGsUKWMXFRlFCkxyG5XiHHNF1wYFKtFZHYU/xRfSPSXXpWIID04cffggAWLhwIe69917s2LEDOTk5eOihhxq2OXXqFMaPH48lS5Zg27ZtLhVKvsmZO+RIOG8ISkLZ+mxCgpSjoYmIvEc7TSw2FJzETVFJCA4UPrAm+CfDvn37EBkZiZEjR9rcpl27dvjkk0+Qm5uLN954Q+ipiADI4xly3qa0oM6nw1JzzJ/d2TZwZPiVc9+IvEeAQoEO4XHYXnTOteMI3bG4uBhJSUkNc5MCAwMBANXV1U2269KlCzIyMrBu3ToXyiS54ZIC8ubPQckWZ9rEF+esEfmz1mFROFF+FXpjveBjCA5M4eHhTSZ4a7VaAEBeXp7FtiqVCpcvXxZ6KiJyEIOSfY62kb3Q5M5eJleeI8f5S0SWFAoF2miicfCaZUZxlODAlJiYiIKCgoa/t2/fHgCwefPmJtsVFBTgzJkzCAkJEXoqIrKDQcl5bC8i/5KuboHfdPmC9xccmPr06YPi4uKG0DRixAiYTCa8+uqrWLJkCY4fP45ffvkFEyZMgF6vx4ABAwQXSUS28cIvnL2249Acke8IVAQgQhmMy9VlgvYXHJjuuOMOGI1GrF+/HgDQs2dPjB07FtXV1fjLX/6CAQMGYPTo0fjtt9+gVqvx0ksvCT0VEdnAsOQ6T7ahu9dgcvdwHJG3S1dH46DukqB9Bd9WdPvttyMvLw8qlarhtYULF6JDhw5YsWIFLly4gNDQUAwYMAAvvfQSOnbkuDeRmHwhLKm+P+PS/voRbUWqhIj8QWywGodLhQ3LuXQftlqtbvL3wMBAPPvss3j22WddOSz5Ca7BJJw3hSVXQ5EzxxYaoEoL6myu2+TptZlcmfDtbpzwTd5OoVBAFRCISoMeaqXK/g6NCA5Mubm5CAkJQWys/S7gq1evoqamBsnJyUJPRzLCJQWkJfew5M6A5Oi5/bHnicNxRI6JVqlxqboU7cKd+54R/GtT165dMXnyZIe2ffjhh9G9e3ehpyIiL6D6/oykYakxudRBRPLTQhWKvKpSp/dzqZ+58TpMYm5L5CmOPkiVj8uwTU5BqTG51eTIhG85D8cR+QqNMhjX6qrtb3gDj1wFqqqqEBTERwkQ+Rq5hZIbOVOfWEOdUj941104f4l8RVigCuWGWqf3c3tgKiwsxMmTJxEXF+fuUxGRB8k9LInFVu+io72TnsD5S0SOC1AoUG9yfo01hyd9f/7551ixYkWT17Kzs3HPPffY3Ke6uhonTpxAdXU1Bg8e7HRxRGJoG16HM+W+08OpjQ+S/cRvOVF9f8ahSeC27pITA4fjiLyfw4Hp4sWL2L59e8PfFQoFysrKmrxmS2ZmJv72t78Jq5BIJqKjA2Sz8jNDExGRMLVGA0IDnf8FyeHANHz4cKSkpAC4PoH7ySefRNu2bW2uuaRQKBAWFoa0tDR07drV6cJInuS8pED3aAMOFbu0tJhXkTI0+ctwnLOkmL/E4Tgi51QZ9IhQBju9n8NXly5duqBLly4Nf3/99dfRuXNnPPjgg06flEhO2kTUOfzkeTn1MgH/G0Zib5NnOTN/icNxRPJSUleNliERTu8n+Nfxo0ePCt2VyKvJLTQBHKJzF2eWk/DVu+OIfE2JvgrdIxOd3o+LyxD5CG18kFsnLjfmaytpO9NuvDuOyLuV6qsRH6xxej+/CUxdunRBZGSk1T/Dhw+32L62thZz585Fz549ER8fjw4dOuDpp5/G1at8/hnJeyFLT4UmbyE03Inxb8zhOCJ5KTfUQqsKhUKhcHpf/5khCyAiIgKPP/64xevmyexmRqMRDz74IDZt2oRevXph5MiROHv2LJYtW4atW7fi559/RkxMjKfKJhHYW1rAmXlMZuYLqtyG5wDObfI0DscReYezFcXoFZUkaF+/CkxarRYvvPCC3e0+//xzbNq0CQ888AAWL17ckESXLFmC6dOnY9asWZg3b56bqyVvIcc5TWbuDE76EW1lf7ec0PWXxFis0t29SxyOI3KO0WTClZpy3Neqs6D95TuuIKFly5YBAP7xj3806bZ7+OGHkZqaii+//BLV1c4/h4bkzZW5KdHRAQ1/5IjDdO7D3iUi73C+sgSdIxIQIGA4DvCzwKTX67F8+XK8/fbbWLRoEfbv32+xTU1NDfbv34+MjAyLoTqFQoFbb70VlZWV+O233zxVNnkZuYYnd0wKl/Pkb7n3LhGR55hMJpyqKELf6BT7G9vgV0NyBQUFmDZtWpPXevbsiX//+99IS0sDAJw7dw5GoxHp6elWj2F+/ezZs+jfv797CyanNbd4pSOPSBEyl6k5joQmTw/n+cMSBJ56FIorvUuc7E3kOWcqi9FFm4DgQOGxR/CeUVFR0Gq1OHnyJIKDnV8x09MmTpyIfv36ITMzE2q1GmfOnMGCBQuwatUqjBw5Ejt37kR4eDjKysoAXJ/vZE1ExPXFrszbNaempka8DyAD2phzUpfgEWKHJnuaC1XuClNihiZvmMvkKG/pXZJi/pKcfp7p9fom//U3ISEhLu1fV1cn6OGz3qrOWI/TZVfxcHJPq1/Hjran4MCkVqvRpk0brwhLAPDXv/61yd+7du2Kjz76CACwatUqfPLJJ3jyySdFPWd+fj7q6+tFPaaUtBLfGNhBa8CJUr/qFLV6ARcrRIk5IVxOoUns3iVbYcnfepdyc3OlLsFCQUGB1CVIIiMjw6X9i4quos6Hrk32nKgvReeACFzOu2T1fUfbU/DVp3Xr1tDpdEJ3l42HH34Yq1atwp49e/Dkk0829CCVlpZa3d7cs2TerjmJic6vJCpv3t/D5MiwHOD5XiZnNA5RYoQnXxmic3Q+la2w5Kl1l7xVcnKy1CU00Ov1KCgoQHx8PFQqldTleJ2YmFi/6WEqrK2AusaAfi07uHwswYFp9OjR+Oc//4nDhw+jW7duLhcilejoaABAVVUVACA1NRUBAQHIycmxur359TZt2tg9tqvdpiSMWA/hlXNoMhMrPIkRmqTsZXJXWHJH75K3kuPPM5VKJcu65C4oKAiBMEldhtvVGetxtKQQf07rjZBA13+WC/6V6qmnnsLNN9+MSZMmWb3bzFuYazffERcaGoqbbroJp0+fxsWLF5tsazKZsGXLFqjVavTo0cPjtZI4nLnYtYmok9WjMJrj6p15YkyCluKuObmEJUd7l7xxOA4AgjXHpS6ByCm7Sy7izvj2CBUhLAEu9DA9++yzSElJwcGDB3HHHXegffv26NChA8LCwqxur1Ao8P777wsu1BWnTp1CUlKSRW2nTp3CK6+8AgB44IEHGl6fPHky9u3bh//7v/9rsnDlf/7zH5w/fx5//OMfERoa6rH6SXre0Ntk5soK5GL1NAFwe2+TM+HMncNwRCQ/p8qvIiFYg4xw8SbfKnQ6naB+uaioKCgUCphMju2uUChQUlIi5FQumzNnDhYuXIj+/fsjOTkZYWFhOHPmDDZu3Ii6ujpMnz4d//jHPxq2NxqNGDNmTMOjUQYMGICcnBysXbsWKSkp2LRpk18+GkXs3zCvVDn/XD5HJ307MiTnyFwmW7wlPAHCgpNYc5rcEZqc7cVqrufMG3uXpFzhu7aio2Tnbqympga5ublITk7mkJwA5ypKYPThIbni2kocLb2Ch9N6CV6k0hrBPUwzZ84UrQh3GzRoEE6dOoUjR45g165dqKqqQnR0NG6//XZMmTIFQ4cObbJ9QEAAPv/8c7zzzjtYtWoVFi5ciKioKEyaNAl/+9vf/DIseRux5jHZ0viCKvfwJOTRLWJNBBezt0nIcJ+cwhIRuV9NfR32XsvFw6nihiXAhR4m8j9y6GEC5NPLJITU4UrK3iYzR8OTq/OhxAxLgDx6lwD2MAHsYXKVr/Yw1ZuM2Fx4BiNadkRSWKTox/evRW3IrzjSy+ToMgNicWYCuTvClZS9TWaemBgut7DkK4I1x2UTmohutKfkIvpFt3ZLWAJEDEwmkwklJSWoqqqS1XodRPZ4OjQ56sYLuFgBSg6hyV2EBCVAWFhyhrfeGUfkLbLLriAuWINuke5b/9DlW0R27tyJcePGISkpCRkZGejevXuT9+fNm4dp06bh2rVrrp6KyG28YV0d8xIHYixzIGQJArEf3Cs2T4clf+tdIpKr3KprKDfocUd8O7eex6XANH/+fNxzzz3YsGEDqqqqYDKZLO6a02g0WLFiBX766SeXCiXpeWNXvDMXNW8ITWZiBidnaOODZBec7NUkdVhi7xKR+xTXVuJURRHGJnVtWALIXQQHpp07d+Lll19GSEgIZs2ahSNHjqBPnz4W240YMQImk4mBiSwInbzqzgtQ2/A6vwtOQtYikkNwcqQGqcOS2KSc8E0kN5UGPfZey8XElJ5QBgS6/XyC5zAtWLAAAPDuu+82LPpoLd0lJCSgZcuWOHLkiNBTEblEyBID5gumHOc2WWO++Aud5yR0sUsxH+Dr7DmbYy8EunvOkpkv9i5x4jfJQZ2xHr8WncPYpG5QKz3zPEHBPUz79u1DVFRUkxWybUlISEBhYaHQUxG5TGivgLnHyVt6nVztcRK68rW7e5zMx5c6LHHeEpH0TCYTthedwx3x7RAXovHYeQX3MOl0OmRmZopZC5Gs3XgxlXPvkyuPchFyF53ZjYFGaM+TkPDlSNhzJUw6G5Z8sXeJSA4O6PLQWZsg6mNPHCE4MEVGRiI/P9+hbc+dO4fYWI69k7TEXv1baK+Tp4KWK8N0rjyPrjFPzHMSIyiJ3bPEsETkHmcqihASEIS+0a09fm7BQ3I9evRAUVER9u3b1+x269evh06nQ+/evYWeisiC0AuSHIZUGg/zeWLIz9UhOrk+oNbR2jwdlojIPa7WVuBSdSnuSZRmdEvwT8KJEyfCZDLhmWeewaVLl6xuc+rUKUyfPh0KhQKTJk0SXCSRP3BngBLjTjq5BCdnapEiLLF3iUh81fV1OHAtD+OTe4j+jDhHCR6fGDlyJO655x6sXbsW/fv3x+233468vDwAwOuvv47ff/8dGzZsgF6vx9ixYzFkyBDRiiZyhbsfzCsWse/Uc/VOOkC8oTqh53WUIwHR23qWuKQA+Svjfyd539eqM8KU0s0ddenhu3q9Hi+88AKWLl0Ko/F/P0AVCgVMJhMUCgWysrLw5ptvIihIvhNkyXFyeQAv4PhDeG3xhtDUmJhzn8R8Tp27wpOQHi1He9LcFZbc2bskp8Ak9bICfPiua7zt4bsHruUhVR2F3i1SJK3DpcBkdubMGaxevRrHjh2DTqeDWq1GZmYmRo0axTvpfIwvBSbA+0ITIF5wcsfDfc2cDVFiDPeJ0asEyDMsAQxMjTEwucabAtOl6lJcri7DuJTuUpcizsN327Zti+eee06MQxF5lPni6E3BSayHBYsxRGeLJ+c7idWrBMg3LBH5o5p6A46WXsaUNMuniEhBHrM4iSQmt/kq9og5MVys59J5mjN1uzMsEZF77Cm5gHtaZiI4UB6/0MqjCvJLrgzHuYM/9zYB7u1xEouzwc7RUOlKWPJE75KchuOIPOF8ZQligzVIUUdJXUoDh64M3bp1c/lECoUChw4dcvk4RO7mbcFJzNAEyDM4CekB85WwRORv9MZ6nCgvxCPpfaUupQmHrggXL150+UTWHsxLJGfWLqRyDVFihyagaUjxdHhyZYjQE0EJYFgicpdDuksYFpeBoIBAqUtpwqGf/mvXrrX6+v79+zF79myo1WpkZWWhQ4cOiIuLQ2FhIU6cOIFPP/0UlZWVePHFF3HTTTeJWjiRFBy5yEoVqtwRmsxuDDBiBygx5lA5M6fLm8ISh+PIn5TW1UBvrEeHiDipS7EgeFmB7Oxs3H777Rg8eDA+/vhjqNVqi20qKysxdepUbN26FT///DM6dpT2VlRyja8tKeBJngxRcn4osDs4O/ndm8ISIM/AxGUFvJuclxX45epZjEzMRGywRupSLAi+S27u3LkwmUz48MMPrYYlAFCr1Vi4cCFMJhPmzp0ruEjyPXKb8O1u3aMNHrsLy53PpZMTZ+8UFOPfgMNw0ocl8l3FtZUIVwbLMiwBLgSmXbt2oUOHDtBqtc1uFxkZiQ4dOmDnzp1CT0XUhLf1LjXmqeDky6FJyJIKYrS5FGFJjr1LRO5yrOwKhsVlSF2GTYKvPGVlZbh27ZpD2167dg3l5eVCT0U+xt96l6zxxPPsxH4WndSEhECxwil7lojcq9xQi6CAQEQHh0ldik2Ce5jS0tJw4cIFbN26tdnttm7divPnz6N169ZCT0XUwJt7l6Tizb1N5t4kfwxLcuxd4nAcucup8qsYEJ0qdRnNEhyYxo8fD5PJhEmTJmHRokWorq5u8n5NTQ0WL16MyZMnQ6FQYMKECS4XS9IRa8I3e5f+x5MrS3tTaHIlJAHiDnsyLBG5n9FkQlFtJdLULaQupVmC75IzGAx44IEHsHXrVigUCoSGhiIlJQWxsbG4evUqLl68iOrqaphMJgwePBhff/01lEr2DngrOQQmX+xdkmIJArkN0YkV5sQMoFIPwck1MMmlh4l3yblGbnfJXaouRU29AXcktJO6lGYJ/mmtVCrxxRdf4I033sCiRYtQXl6OEydO4MSJEw3bhIeH45FHHsGMGTMYlohhSSYaBxRPhyd39XQxLLmfXMIS+Z7cKh1uj5d3WAJc6GFqrKqqCrt378apU6dQUVEBjUaDdu3aoW/fvggLk+8ELnKM1L1LvhqW5LhquKsBytNDf2IPazIs2SanwMQeJtfIrYdpQ8EpPJbeV/ZPBBHlJ3ZYWBiGDh2KoUOHinE48kEMS03JMSwB3jPXyR3zv6QOS3Imp7BEvqXSoIc2KET2YQkQKTCR7xKjd0muYalxaPHkBGwSxl3/RnIJSnLuXSJyl6u1FUgNi5K6DIe4fEUqLy/HsmXLsGHDBpw+fbrJkNydd96Jhx56COHh4WLUSl5I6rDkaE+Ore3EvkjLtWdJztwZZhmW7GPvErmTrq4GbTTRUpfhEJd+eh88eBBZWVnIz8+HyfS/8dDy8nJcvnwZ27Ztw4IFC/Dpp5+iR48eLhdLniX2s+Mc5WpYEjOU3HgsoRdvBiXn+UNQAhiWyL9VGGoRrbL+eDW5EfxTvKCgAGPGjEFJSQnCw8MxadIkZGZmIiEhAVeuXEF2djY+++wzXLp0CWPGjMGOHTsQHx8vZu0kc872LrkSlDwVSKydx9qFnQFJGE8MjTIsEclHdX0dNEqV1GU4RPBP9ffeew8lJSUYMmQIli5disjISIttnn/+efzxj3/E1q1bMX/+fMyaNcuVWsmDXO1d8kRYkksokUsd3szfghIg/7DE3iXyBAUUXjHhG3Bhpe+NGzdCpVLh448/thqWAECr1WLRokVQKpXYsGGD0FORl3F3WDpUrGRI8QHmFbkZluSHYYnIkuCrTl5eHjp27IiYmJhmt4uNjUXHjh1x+vRpoaciD3Old8mdYYkhyft5+m5EuQUlQP5hiYisc2ml79raWoe21ev1XOnbS3hyorejYYlBybtJtWQDw5Iw7F0iT1IoAIPRCGWA4AEvjxF8JWrTpg2OHDmCkydPon379ja3O3HiBE6ePIlu3boJPRV5CWd6lxwJS0KDkr3Vqr1lcUZvx6D0P94QlACGJfI8TWAwSuuqER0s/zvlBEe6kSNHwmg0YtKkSTh06JDVbQ4dOoSHHnoIAHDvvfcKPZVL8vPzsXDhQtx3333o3LkzYmNj0a5dO0yaNAn79++32H7OnDmIjIy0+efChQsSfArP8MRQ3IlSpahh6Ux5kMUfd+xDjvPUvKQbddAaGJZcwLBEUogICsblmnKpy3CI4B6mRx99FKtWrcLJkycxdOhQ9O3bF5mZmYiLi0NhYSGys7Oxe/dumEwmdOzYEY8++qiYdTts0aJFmDdvHtLS0nDrrbciJiYGZ8+exQ8//IAffvgBH3/8MUaPHm2x34QJE5CSkmLxular9UTZHuepsGSPvaDkrnDT+Lhy74Fqrg2krF3K1dLlGJQA7wlLRFKJDdbgfGUJOmsTpC7FLsGBKSwsDN999x2mTJmCHTt2YNeuXdi9e3fD++aFLAcOHIjFixcjNDTU9WoF6NmzJ77//nsMHDiwyes7d+7Evffei+nTp2P48OEIDg5u8v6DDz6IQYMGebJUryRWWJIqKNk7l5QBRMhnvnEfT9XP4bemvC0osXeJpNJCFYaDuktSl+EQl2bTJiQk4Pvvv8euXbusPhrljjvuQN++fcWqVZCRI0dafb1///4YNGgQNm/ejOzsbL9eiVxo75InwpLUw2WeDE/u+KzmY7qzdqmG3+SKYYnIcQEKBcICg1Cir0ILVZjU5TRLlNuP+vXrh379+olxKI8KCrp+MQkMDLR4b+fOnThw4AACAgKQnp6OW265BRqNxtMlup2UYUnMoHS2zPb2bSLECQti9t54OgieKQ9yS2jiMgH/421BCWBYInlIV7fAnuKLuKtlB6lLaZbf3q+dm5uLX375BQkJCejUqZPF+3PmzGnyd61Wi9dffx0TJkxw+Bw1NTUu1+luwQIyoDvDkiNBorlw5Oj2YoQoqXu/nCV2aGJY+h9vDEulRWkA5P8zyhq9Xt/kv/4mJCTEpf3r6upQbzKKVI3rogNCcKg8H+XaSgQFWHZguJuj7anQ6XQm+5tZOnfuHL744gt0794dd955p83t1q1bh8OHD2P8+PFo3bq1kFOJrq6uDvfeey927tyJDz/8EOPHj294b+3atSgtLcXAgQORkJCAgoICrF+/HrNnz0ZpaSmWL1+Ou+++26Hz5OTkoL6+3l0fw2Wdewi7ADkSmJwNS82FD2cDkrPE6oHyFmKEJk+GJQYl9zj2m9/+vuz1MjIyXNp/5+ljqJPZtelSfRWCFQHoGtTC4+d2tD0Ff8csW7YM7777Lj777LNmtzMajZg7dy4MBgNeeukloacTjdFoxBNPPIGdO3di8uTJTcISANxzzz1N/t66dWs88sgjaN++PUaNGoVZs2Y5HJgSExNFq1ts2phzgvZzJSzJLShZO48/hCd3Dc+JTc5BCfDusFRalIbkZKmrEE6v16OgoADx8fFQqbzjwa1yEhMTK6seJgCINZmwqTgHQxMTEBooz957wT1MQ4YMwenTp5GXl4eAZlborK+vR1JSEjp06IAtW7YILlQMRqMR06ZNw4oVKzB27Fh8+OGHzdZ+o549eyInJwcXL15ERESEGyt1PyFzlzwVlpwJSsXF1r/po6NdWzXWH4KT0NDkj89+a8ybgxLgG/OWampqkJubi+TkZJeHp/zRuYoSGCHo0u9Wl6vLcLmmHGOSu0pdilUuPUsuOTnZbuAIDAxESkoK8vLyhJ5KFOaepZUrV+KBBx7ABx984FRYAoDo6Gjk5OSgurraqwOT1GFJaFCyFY4c3daZEOVvvU5ywaDkXr4Qlsh3tQyNwPmqazhVfhXtwuX3/SY4MFVWVjo8J0mtVuP8+fNCT+WyxmFp9OjR+Oijj6zeGdecyspKnDhxAmq1GtHR0W6q1P3kGJaaC0rOhCR7Gh9LSHjyteAkt6E5uYYlXwhKAMMSeYebo5KwseAUWoZEIDwo2P4OHiQ4MMXExODMmTOor69vNnzU19fjzJkzkoUM8zDcypUrMWrUKCxatMhmveXl5SgoKEDbtm2bvF5dXY2nn34a5eXlmDhxol89SFhoWHJkCM5WUBIzJNkiJDx5qtfJE0skyAmDEhGZBQUEoneLFKzMPYQ/pfVCoEI+D+UVfOXv3bs3vvvuO3z22WeYPHmyze2WL1+O8vJy3HrrrUJP5ZK5c+dixYoV0Gg0aNu2Ld58802LbYYPH46uXbuipKQEvXr1Qs+ePdGuXTvEx8ejsLAQW7duxaVLl5CZmYnXXntNgk8hDlcef2KLI2HJ0V6l5oJSaYHwoKCNd2yoT+iQHeB4kHF1IrvYSyQ428sk9vwlhiXPYO8SeZMWqjCkqVvg27xjuD+pCxQKhdQlAXAhMD388MP49ttv8de//hUqlcrq+kQrV67EzJkzoVAo8Kc//cmlQoW6ePEiAKCiogJvvfWW1W1SUlLQtWtXREVFYcqUKThw4AA2btwInU6H0NBQtGvXDo8++iimTp0q2SNeXOWOoTghYcmZoORKSLJ1nObCk5DgZOapO/qaO7e39T7JMSz5WlACGJbIO6WpW+CI4TI2FZ7BbfGuLaMgFsF3yQHA9OnT8Z///AcKhQKJiYm4+eabodVqUVpaiv379yM/Px8mkwkPP/ww/vWvf4lZNznJ2cDkibBkLSiJFZLssdfrBLh+p51UhAQnR3uZxOphkltY8sWgBPhuWOJdcq6R611yNzKZTNh7LRcpoZEYGJsmdTmurfT99ttvIzY2Fu+99x4uXbqES5eaPkAvNDQUzzzzDGbMmOFSkeQabwhLQoKS6vsz0I9oa39DK8zns9fj5I2h6WxZkKx7m+QSlnw1JBH5CoVCgd5RydhVcgHK4gD0jZZ28WuXepjMioqKsGHDBmRnZ6O8vBzh4eHo1KkT7rjjDq++o8wXSB2WXO1VUn1/ptl6muNomPLV3iZnQpOnepjkEJb8JSj5au8SwB4mV3lLD5OZyWTCruILSFO3kLSnSZTARPIlZmASOyzZCkquhCRr/Dk4ORqaHAlM3h6W/CUoAb4dlgAGJld5W2AC/jc8Fx+swTCJ5jR5109/copcw1JpQZ3VsKT6/ozoYcmZ4zoyLFhcbGz44w0cnYzu7gcJSxmWEsJi/SosEfki8/Ccrq4aqy/9DqPJ84HP5QWFdDodPv3004Zb76urq3Ho0KGG99evX4+SkhLcf//9fOaPB4k9FHcjV8LSjdwRkqwxn6e5HidH5jaZOfJYFjGDldDeLannNEkVlvw1JPl67xL5L4VCge6RrXCyvBArLv6GccndoXTyiR2ucCkw7dixA3/84x9RXFwM03/T3o3rJezfvx9vv/02WrRogTvvvNOV05GbCJm31Jg3hKUbz2lvmK60oM6h0GSNu3qfhK5U7m/8NSgBDEvkH9qHx+FC5TUsPb8PD6b0RJjSM8u6CP6pe+HCBUyYMAFFRUW48847sXDhQnTsaPnNOnr0aJhMJvz4448uFUqOE3OBSntDcbbC0o3DVjeGJXcNvznK0SE6Ty1z4CxvGRb0ZO8Sh96I/EdrdRQ6axPwn/P7UKKv8sg5BQemefPmoby8HM8++yxWrFiBCRMmQKvVWmzXsWNHaLVa7Nmzx6VCyT1cmbfUXFgysxY6pAxKjTlah1xDE+B4b5YUC2t6KiwxKF3H3iXyN7HBGgyITsXnF39DbuU1t59PcGDavHkz1Go1XnjhBbvbpqSkWKzRRO7hTO+SK/OWHA1LjUndq2SNM6FJrsHJEz1Nzt4h58mwRET+KzwoGENj2+KHKydwtPSKW88lODBduXIF6enpCAqy/5trcHAwamtrhZ6KHOTOoThbYckWufYqWeNMbf4cmuSEvUpNsXeJ/FlwoBJD49riQEkufr16zm3nERyYQkNDodPpHNr28uXLiIyMFHoqcgNnh+Jssda75E1hyczZ0CTnHiepubN3iUGJiKwJVARgYEwaLteUYW1+dsONaGISfJdc+/btceDAAeTl5SEpKcnmdr///jsuXbqEYcOGCT0VOUCs3iVX5y15Y1gyE/KoFXeGJmfv0rP3KBeplxdwFYOSdexdIrpOoVDgpqgkHC8rwKrcwxiT3BWBCvHuKBZ8pHvvvRf19fV44YUXUF9fb3Wbmpoa/OUvf4FCocB9990nuEgSlzNzl5ydt9SYN4UlMznVLJdeLGfmL7mjd4m9SkTkjI4R8YgODsPyC7/BYLSeT4QQHJj+9Kc/oX379vjhhx9w5513YunSpSgrKwNwfX2mjz76CIMGDcLu3bvRtWtXjBs3TrSiqSmxJno7OhTnyCRvOQUPb+eJ0OTos+Sa466wRLaxd4nIunR1NJLDtPjswm8wGMWZ4yl4SC4kJARfffUVxo8fjwMHDuDgwYMN791zzz0Arj/7JTMzE59//jmUSpcXFScPc2aity+FJSFDc+7mykKa3ophiYhckRIWBZMJWH7xIB5q3dPl4TmX9k5KSsKWLVvw7rvvYsiQIYiOjkZgYCAiIiLQt29fvPHGG9iyZQsSExNdKpJsc1fvkivzlsg9pGhnR4fjxOxd4hCcY9i7RGRfa3UUEkMj8G3eMZcngrvc7RMUFISsrCxkZWW5eijyIr40b8kaOfYyucobJnwzKBGR2NLV0Thcl4+tV8/iljjhP9f5QCovJmXvkpkvDcWRMGL1LjEsEZG7dNW2xNnKElx0YUVwBiZyCIfi5MEdbW5rwrezq3u7gmHJORyOI3KOQqFA3xatsfZyNuoE3jnn0JDctGnTBB28MYVCgffff9/l49B1nu5dsoe9S/5JjN4lhiUi8oSQQCXaaWKxo+g8bolr4/T+DgWmzz//HAqFwuqEKYVC0fD/jd+/8XUGJu/lr71LcpzHJPROOVvzl1zpXWJYkgZ7l4iES1O3wPqCU+gf0xqqAOemcTu09cyZM62+rtPpsGTJEtTV1aFnz57o0KED4uLiUFhYiBMnTuDgwYNQqVR4+OGH+WgUGWLvEkmJYYmIPE2hUKB1WCSOlxWiW6Rzd/A7FJj++te/Wrym0+lw6623Ijk5GQsXLkSfPn0sttm7dy+mTZuGDRs2YPPmzU4VRraJNRznLH97wKvcepfsae6xKM7wRO8Sw5Iw7F0icl1yWCR+LytwOjAJ/gk7e/ZsXLx4EStXrrQalgCgd+/e+Pzzz3H+/Hm8/vrrQk9FbtDcqt68M863ODscZw/DEhF5M40yGJUGvdP7CQ5M69atQ/v27ZGRkdHsdhkZGejQoQN+/PFHoaeiRtzVu9R4OM4af+tdkqPm5i95snfJFQxLwpUWpUldApHPELKIpeCfsoWFhQgIcGz3gIAAFBYWCj0VScRe75Iv87bhOFvk1LvEsEREcmAymSBkzW/BgSkmJgYnTpzA+fPnm93u/PnzOH78OGJiYoSeikTm6HCcPb46HCfHsOTtvUsMS6459hufxUkklnJDLVqoQp3eT/BP2j/84Q+or6/HxIkT8fvvv1vdJjs7Gw899BBMJhPuuusuoaei/5JqOM6feFtYao4zvUvunOjNsEREcpJTWYwu2pZO7yf4SvnCCy9g3bp1yM7OxqBBg9C/f3907NgRsbGxuHr1Kk6cOIEdO3bAZDKhVatWNpcmIPnzl/lL3hiWbPUuyWUojmHJddfnLuVKXQaRTzAYjSioqcB9rZz/2SQ4MEVHR+PHH3/Eo48+it27d2PHjh3YuXNnw/vmCVV9+vTBhx9+yCE5kjVfCkvOste7xLBERL7ioO4SBsemI6DR4tqOcmksJiUlBT/99BN27dqFjRs34tSpU6ioqIBGo0G7du1w2223oX///q6cgkTW3PylxqxN+PZVcgtLjgzBNReWxB6KE4JhSRzX112qkboMIp+QX10GwITO2gRB+4syeaVfv37o16+fGIciGzyxWKW9Cd++eIecP4clRwjpXWJYEgcXqSQST4m+Cr+XXcEfU3sJPgZn+/o5ZyZ8a+ODfCo0ySUsOTOpW8yw5I6hOIYlIpKb0rpq7C25iMmpNyMkUHjsYWAivyOHoCTkzje5hyUSD3uXiMRRUFOOQ7p8PNT6JmiUwS4di4GJ/IqQsCT0tn6x2JvcLZewxN4lcTAsEYnjXGUJLlRdw5/SeiEk0PWf4wxM5BecCUpSByQzR+6CEzssCcWwRERyYTAase9aLsKVwfhj6s0IVIhzRzEDkxdwZsK3J+lHtJX9at+OBiW5hCTAtaAEuBaWOG9JWuxdInLNNX019pRcxC2x6egk8G44WxiYyGc5Epa8LSgBDEu+imGJSDijyYSjpZdRWleDh1r3hDYoRPRzMDCRXdHRAQ2rfd94p5wce5nsBSVvDElA80EJYFjyZgxLRMIV1VbiwLU89GqRjPuikqAQsCilIxiYmnHw4EHMmTMHe/bsgcFgQGZmJqZNm4b77rtP6tJsEroG043aRNR53eKVcg9KrqzMzbDkuxiWiISprTfgoO4SAhQKTGrdE+Fu6FVqjIHJhm3btuH+++9HSEgIRo8eDY1GgzVr1uDhhx9GXl4ennrqKalLFF3b8Dq7i1daI3Uvk1hBSaxHjYhJaFACGJa8AcMSkfNMJhNOVxThXGUJ7kxojzaaaI+cV6HT6UweOZMXMRgM6NWrF/Lz87Fx40Z07doVAFBaWophw4bh4sWL2L9/P1JSUjxSj5irfFt7NErjxSsbB6Ybe5gaP4TX2gKWngxNYoQkOQYkM3tBCXD9TjiGJWk5GpZqamqQm5uL5ORkhIS49zdob8E2cc25ihIY4Z2X/svVZThSehmdtQkYEJMq2h1wjpDvFUNC27Ztw7lz5/DAAw80hCUA0Gq1mD59OvR6PVasWCFhhbaJeUFr7qJtLZDoR7R1+6KQ9s6hjQ9qNixFRwc0/JGbNhF1DX+a0za8zuUhOIYlabFnicg5pXU1+KXwLIr0lfhj6s0YHJvu0bAEcEjOqu3btwMAhg4davHesGHDAAA7duzwaE2e0tywXOPJ34DtR6WYA40YPU5iLAsgx3AEONaL1Ji958G5awgOYFgSE8MSkeOq6+twSJcPo8mIka0yERuskawWBiYrzp49CwBo06aNxXvx8fHQaDTIycmxe5yaGnGeMi7h14fF5G9HQxPQNOw4Ep6c7Z3ylpDkbDC6kRhBCeAQnByUFqUBcO7ngl6vb/JfYpu4OgxZV1eHepPR/oYSqjPW43jFVVyrq8bQ6HSkhEUCJvGuq4052p4MTFaUlZUBACIiIqy+Hx4e3rBNc/Lz81FfX+9yPdoYlw/RoIPWYDGPqXu0ock8pht7mRwJTYD1eU1mYg3ViRmSXA0y7iZWUAIYluTg2G9KALmC9y8oKBCvGB/hr22SkZHh0v5FRVdRJ8K1yR2MJhNyjZW4aqpB18Ao9AiMgaK4HLnF5W47p6PtycDkRomJiSId6ZxIxxHPjaEJcCw4Ocve5G0x1zGSA3shycydvUoAw5LYSovSkJwsbF+9Xo+CggLEx8dDpVKJW5iXYpu4JiYmVnY9TCaTCeeqryGn6hpuapGIByJaIsBN6ykJxcBkhblnyVYvUnl5OSIjI+0ex5vu3nC2lwn4X1ixFZwA58KT2Lf/e0NAAhwPSYD7e5UAhiUxmecrifGjQKVSedXPFE9gmwgTFBSEQJncJWcymZBbrUN2WSE6RyTg8cQOUAYESl2WVQxMVpjnLp09exbdu3dv8l5BQQEqKirQs2dPCSoTh7VhOWscCU2A7eAEiLdYpK+EJGfCkZmzD8wVGpQAhiUxcXI3UfOu1JTjaOllpIZF4c9pvRASKO/FkhmYrBgwYAD+9a9/YfPmzbj//vubvLdp06aGbeQqISxW0IrfN/YyAdZDE2C5RhPQNNRYC0+O8tQwm5Dw4knOBiWAvUpywKBE1Lxr+moc0l1CC1UYHmrdExplsNQlOYQLV1phMBhw88034/LlyzYXrty3bx9at27tkXqcWbjSzJHAZKuX6cbQBMDmUgOefnyKkIAk92DUmJCQBLBXSS7cEZa4SKMltolrpFq4ssqgx2+6fAQoFLgroT2ig9Uer8EV7GGyQqlU4r333sP999+P4cOHN3k0Sm5uLl577TWPhSUpONLTZNZcj5MYxF6rSG6EBiQzV4ISIF1YshUshPxyIAfsVSKyrc5Yj2OlV3Ctrhp/iG+HZHWU1CUJwh6mZhw4cABz5szB3r17UVdX1/Dw3dGjR3u0DqEXEVd6mQDrPU2A7d6mxoQEKPYeOc7bgpLQx4DIPUB5IiixN8US28Q1nuphMplMOFNZjLMVxRgSm45OEfFQyOzON2cwMHkJdw3LAcJCE+BYcBKbqwFJzNAiBVeDEuC5sCQkTDR3IZRLePJ0bxLDgSW2iWs8EZgKayrwm+4SOkXEY2BsmscfY+IOHJLzYUInfzdmDhjWgpM5vLgrOAkNR94eim4kRkgC5B2UnD2uFOGJw25E9lXX12H/tTyoA1XISr3JayZ0O4KBiRxaZsDavCYzW8HGkSDl7z1GzWFQcuxc7gpPDEhEjjOZTDhZcRW5VToMT+jgtfOUmsPA5OMc7WUyX5ybC06Nw0lzQ3VmYs8v8uVwBIgXkBrzRFiSOlhYO7+zIUrqz0Dkza7pq7CvJBedtS3xSHpf2a3QLRYGJi9RW9HRI8MQji5q6Wx4cpavhyPAPQHJzBd7lZwh17qIfInRZMLR0ssoravBg617Qhvk2/PJGJj8gLNzmRwNTWb+EG5c5c5w1Ji/ByUi8gydvhp7Si6id4tk3BSV5NV3vzmKgclPCAlNQPNDdHLnqZAiBwxKROQJ5rlKl6vLMNEPepUa896roR9ydVhOyF1zjUOH3MKTPwUiWxiUiMhT9EYDdhZdQJo6Cn9K6+2zc5VskdcVkNzOlaUGpAhPDEWW5LroJBH5rmv6auwuuYDhCR2RpmkhdTmSYGDyMmJM/hZjfSZrQUZoiGIocgyDEhFJ4WLVNZyuKEJW65sQ4UdDcDdiYPJT5ouvq8GpMQYf8UnxrDcGJSIyO1FWCF1dNf6U2gvKgECpy5GU969V7ofEvKDxSfXykxAW2/DH0xiWiAi4Prn74LU8mGDChJQefh+WAPYweS0x12VyR28TOU4OoZVBiYjMTCYTDujyEK0Kw23x7aQuRzYYmLyY2ItZMji5nxzCUWMMSkR0o4O6S4hRqTEsPkPqUmSFgYksNL6oe3N4shZO3P155BaIbGFQIiJrjpcVICwwiGHJCgYmL+fuR6bcGACkCFBihhBvCTTuwqBERLbkVulQZqjFhOTuUpciSwxMPsBTz5kD7AcOe4HK3wOLVBiUiKg55XU1OF5eiClpvf3iMSdCMDD5CE+GpuYwEMkLgxIR2WM0mbCz+ALGp3RHEO+Gs4mByYfIJTSRtBiSiMgZR0svo1eLZLRQhUldiqwxMPkY88WSwcn/MCgRkbNK66pRWleD+6KSpC5F9rhwpY/ixdN/1FZ05L83EQly8Nol3JOYyXlLDmAPkw9jb5PvYkAiIlddrilDTLAaMcFqqUvxCgxMfoDByTcwJBGRmLLLCriEgBMYmPwIg5P3YUgiInco1lchKigM4UEhUpfiNRiY/BCDk3wxIBGRJ5wqv4phcW2lLsOrMDD5scYXZ4YnaZQWpSEkhL/hEZHnGGFEhaEWLUMjpC7FqzAwEQDLng0GKPE1buOamhrk5uYiOVnCgojIL12uLkOH8Dipy/A6DExklb2hIU8FKrnU4QgOpxGRN7hUXYbbE9pJXYbXYWAiQZwJB//rTUkWffiJIYWIyDlV9Xqu6i0AF64kIiLyIwxLwjAwERER+ZGUsCipS/BKDExERER+JClUK3UJXomBiYiIyI9E81EogjAwERER+ZEAPmhXEAYmIiIiIjsYmIiIiIjsYGAiIiIisoOBiYiIiMgOBiYiIiIiO3w6MNXV1WH16tV47LHH0Lt3b7Rq1QpJSUkYNmwY/v3vf6O+vt5inwsXLiAyMtLmnzlz5kjwSYiIiEhKPv0suXPnzmHy5MnQaDQYPHgw7rrrLpSVlWHdunV47rnnsGHDBqxcuRIKK7dYdu7cGcOHD7d4feDAgZ4onYiIiGTEpwOTRqPBW2+9hQkTJkCt/t9CXbNmzcKIESOwfv16rF69GqNGjbLYt0uXLnjhhRc8WC0RERHJlU8PySUmJmLKlClNwhIAqNVqTJs2DQCwY8cOKUojIiIiL+LTPUzNCQoKAgAEBgZaff/KlStYvHgxysrKEBsbi0GDBiEtLc2TJRIREZFM+G1g+uyzzwAAQ4cOtfr+li1bsGXLloa/KxQKjBkzBu+8845FjxURERH5Nr8MTEuXLsXGjRsxePBg3HHHHU3eCwsLw4wZMzB8+HCkpaXBZDLh8OHDeO211/DFF1+guroan376qUPnqampcUf5Xkev1zf5L7FNbGG7WGKbWPL3NgkJCXFpf16bmnK0PRU6nc7k5lpc9tJLLzn1jfHYY4+hTZs2Vt9bt24dJk2ahISEBGzcuBEJCQkOHbOqqgpDhgzB6dOn8csvv6B79+5298nJybG6dAEREZFQGRkZLu3Pa1NTjranV/QwLV26FJWVlQ5vP3LkSKuBacOGDZg8eTLi4uKwdu1ah8MScL3nady4cZg1axb27NnjUGBKTEx0+Pi+TK/Xo6CgAPHx8VCpVFKXIwtsE+vYLpbYJpbYJq7htUkYrwhMly5dcvkY69evR1ZWFqKjo7F27VqkpqY6fYzo6GgA13ubHOFqt6mvUalUbJMbsE2sY7tYYptYYpsIwzYTxqeXFTAzh6WoqCisXbsW6enpgo6zf/9+AEBKSoqY5REREZHM+Xxg2rhxI7KyshAZGYm1a9fanNtkdvjwYZhMltO61qxZgxUrViAyMhK33Xabu8olIiIiGfKKITmhTp06hYceegi1tbUYOHAgvvrqK4ttUlJSMHHixIa/v/jiizh//jx69eqFxMRE1NfX48iRI9i1axeCg4OxcOFCaLVaT34MIiIikphPB6aCggLU1tYCAL7++mur2wwYMKBJYBo3bhzWrFmD/fv3o7i4GEajES1btkRWVhaefPJJtGvXziO1ExERkXz4dGAaNGgQdDqdU/tkZWUhKyvLPQURERGRV/L5OUxERERErmJgIiIiIrKDgYk8wtZDjv0Z28Q6tosltokltgl5mlc8GoWIiIhISuxhIiIiIrKDgYmIiIjIDgYmIiIiIjsYmIiIiIjsYGAiIiIisoOBiYiIiMgOBiYX1dXVYfXq1XjsscfQu3dvtGrVCklJSRg2bBj+/e9/o76+3ua+X3zxBYYOHYrExES0bt0a48aNw6FDhzxXvJsdPHgQY8aMQUpKChITE3Hbbbfh22+/lbost8rPz8fChQtx3333oXPnzoiNjUW7du0wadIk7N+/3+o+ZWVlePHFF9G5c2fExcWhS5cu+Pvf/46KigoPV+9Z8+bNQ2RkJCIjI7Fv3z6L9/2pXdauXYtRo0YhLS0N8fHx6Nq1K/785z8jLy+vyXb+0CYmkwlr1qzBiBEj0L59e7Rs2RI333wznnnmGZw/f95ie39oE5IHrsPkolOnTqF3797QaDQYPHgwMjIyUFZWhnXr1uHy5cu48847sXLlSigUiib7vfXWW5g1axaSk5MxcuRIVFRU4JtvvoFer8fq1avRt29fiT6ROLZt24b7778fISEhGD16NDQaDdasWYPc3Fy89tpreOqpp6Qu0S1eeeUVzJs3D2lpaRg4cCBiYmJw9uxZ/PDDDzCZTPj4448xevTohu0rKyvxhz/8AUePHsXQoUPRtWtXHDlyBJs3b0bPnj3x448/IiQkRMJP5B7Z2dm49dZboVQqUVlZiY0bN6JXr14N7/tLu5hMJjz77LNYunQp0tLSMGzYMGg0Gly+fBk7duzA4sWL0a9fPwD+0yYvvfQSFixYgISEBNx9990IDw/HsWPHsHnzZmg0Gqxfvx6ZmZkA/KdNSB4YmFyUn5+PH3/8ERMmTIBarW54vbKyEiNGjMBvv/2GpUuXYtSoUQ3vnT17Fn369EFqaio2bdoErVYLADhy5Ahuv/12pKamYteuXQgI8M4OQIPBgF69eiE/Px8bN25E165dAQClpaUYNmwYLl68iP379yMlJUXiSsW3Zs0atGjRAgMHDmzy+s6dO3HvvfdCrVbj5MmTCA4OBgDMnj0bb7zxBp555hm88sorDdubg9c//vEPTJ8+3ZMfwe3q6upw2223ISgoCOnp6fjiiy8sApO/tMsHH3yAF154AVOmTMHcuXMtVq82GAxQKq8/I90f2qSgoAAdO3ZEq1atsH379oafjQCwYMECvPTSS5g4cSIWLFgAwD/ahOTDO6/IMpKYmIgpU6Y0CUsAoFarMW3aNADAjh07mry3fPlyGAwGPPfcc01+IHTt2hX3338/Tp48iV27drm/eDfZtm0bzp07hwceeKAhLAGAVqvF9OnTodfrsWLFCgkrdJ+RI0dahCUA6N+/PwYNGgSdTofs7GwA13sXPv30U2g0GsyYMaPJ9jNmzIBGo8GyZcs8UrcnvfXWWzhx4gTef/99q4+38Jd2qa6uxty5c5GamorXX3/daluYw5K/tMnFixdhNBrRt2/fJj8bAeAPf/gDAKCoqAiA/7QJyQcDkxsFBQUBsHzm0fbt2wEAQ4cOtdhn2LBhACxDljfx9c8n1I1fD2fPnsXly5fRp08fq4G7T58+OH/+vMU8Fm926NAhvP3225g5cyY6dOhgdRt/aZfNmzdDp9Nh+PDhqK+vx5o1a/DOO+9gyZIlyMnJabKtv7RJmzZtoFKpsHv3bpSVlTV5b926dQCAIUOGAPCfNiH5UEpdgC/77LPPAFgGh7Nnz0Kj0SA+Pt5inzZt2jRs463MtZs/S2Px8fHQaDQWFwRfl5ubi19++QUJCQno1KkTgP+1U3p6utV90tPTsWnTJpw9exZJSUkeq9Vdamtr8fjjj6NLly54+umnbW7nL+1ivsEjMDAQAwYMwJkzZxreCwgIwBNPPIFZs2YB8J82adGiBV5++WX87W9/Q+/evZvMYdq2bRumTJmCRx55BID/tAnJBwOTmyxduhQbN27E4MGDcccddzR5r6ysDLGxsVb3Cw8Pb9jGW5lrj4iIsPp+eHi4V38+Z9XV1eHRRx9FbW0tXnnllYYeJnMb3Dj0YGZuP19pq9mzZ+Ps2bP45Zdfmn3SvL+0i3loacGCBejWrRs2b96Mdu3a4ciRI3jmmWfw/vvvIy0tDX/+85/9pk0AYNq0aUhMTMT/+3//D0uWLGl4vV+/fnjggQcahin9qU1IHhiY/uull16CXq93ePvHHnvMag8KcL3reMaMGUhOTsaiRYvEKpG8kNFoxBNPPIGdO3di8uTJGD9+vNQlSWLv3r2YP38+/vrXvzbc4eTvjEYjAEClUmH58uVo2bIlgOvz3ZYuXYqBAwfi/fffx5///Gcpy/S4uXPn4q233sKLL76IsWPHQqvV4ujRo3jxxRcxYsQILFu2DHfffbfUZZIfYmD6r6VLl6KystLh7UeOHGk1MG3YsAGTJ09GXFwc1q5di4SEBIttIiIibP7WU15e3rCNt7L3m115eTkiIyM9WJE0jEYjpk2bhi+//BJjx47FO++80+R9czuVlpZa3d9eT523MBgMePzxx9GpUyc8++yzdrf3l3Yx19+9e/eGsGSWmZmJ1NRU5OTkQKfT+U2b/PLLL5gzZw6eeOKJJl8r/fr1w8qVK9G9e3f87W9/w9133+03bULywcD0X5cuXXL5GOvXr0dWVhaio6Oxdu1apKamWt2uTZs22Lt3LwoKCizmMTU3/8dbNJ6H1b179ybvFRQUoKKiAj179pSgMs8x9yytXLkSDzzwAD744AOLZSLM7WRrPpf5dW/+WgCAioqKhq9rW0PRt99+O4Dr8/7Mk8F9vV0yMjIA2B5SMr9eU1PjN18rGzduBAAMGjTI4r34+HhkZGTgyJEjqKio8Js2IflgYBKJOSxFRUVh7dq1NiciAsCAAQOwd+9ebN68GRMmTGjy3qZNmxq28VYDBgzAv/71L2zevBn3339/k/d84fPZ0zgsjR49Gh999JHVOTtt2rRBy5YtsWfPHlRWVlqs47Vnzx60bt3a6yesBgcHY9KkSVbf27lzJ86ePYu77roLMTExSElJ8Zt2MYeCU6dOWbxXV1eHnJwcqNVqxMTEID4+3i/axDwtwjy/60bFxcUICAhAUFCQ33ydkHxwWQERbNy4EVlZWYiMjMTatWvt/kYzceJEKJVKvP322026k48cOYKvv/4a7du3b1jd1xsNGTIEqamp+Oqrr3DkyJGG10tLS/Gvf/0LKpXKZ+fymIfhVq5ciVGjRmHRokU2JzgrFApMmjQJFRUVePPNN5u89+abb6KiogKTJ0/2RNluFRoaivnz51v907t3bwDA9OnTMX/+fHTt2tVv2iUtLQ1Dhw5FTk6OxXpB77zzDkpLSzF8+HAolUq/aRPzEw4WLlxoMdS2ZMkSXLp0Cb1790ZwcLDftAnJB1f6dtGpU6cwaNAg1NbW4v7770fbtm0ttklJScHEiRObvMZHo/jmo1HmzJmDuXPnQqPR4LHHHrMaloYPH96woGdlZSXuvPNOHDt2DEOHDkW3bt1w+PDhhkc7/PDDDwgNDfX0x/CYxx9/HCtWrLD6aBR/aJdz587hjjvuwNWrV3HnnXc2DDlt27YNycnJ+PnnnxuG7f2hTerr63HPPfdg586diI2NxV133QWtVovDhw9j27ZtCA0Nxffff4+bbroJgH+0CckHA5OLfv31V9xzzz3NbjNgwAD88MMPFq9/8cUX+OCDD3DixAkEBQWhb9++ePHFFy3m/XirAwcOYM6cOdi7dy/q6uqQmZmJadOmNXmWmq8xB4DmLFiwoEmALi0txeuvv461a9c2zGsbNWoUZs6c2bDMhK+yFZgA/2mXvLw8zJ49G5s2bUJJSQni4+Nx11134fnnn7eY8+UPbVJbW4uFCxfi22+/xZkzZ6DX6xEXF4eBAwfiueeeQ/v27Zts7w9tQvLAwERERERkB+cwEREREdnBwERERERkBwMTERERkR0MTERERER2MDARERER2cHARERERGQHAxMRERGRHQxMRERERHYwMBERERHZwcBEJFMXLlxAZGQkIiMjpS6lCbnW5YvY1kTyoZS6ACIiKc2ZMwfA9efaCQkmv/76K7Zv344uXbpgxIgRIldHRHLBHiYickpQUBAyMjKQkZEhdSmimDt3LubOnYvS0lJB+2/fvh1z5861+oBtIvId7GEiIqckJiZi3759UpdBRORR7GEiIiIisoOBiWRvzZo1GDduHDIyMhAbG4uMjAw8+OCD2LFjh9Xt58yZg8jISDz++OOor6/HggUL0L9/f7Rs2RKtW7fGuHHjcOjQoWbPefjwYUybNg3du3dHQkICUlJS0L9/fzz//PM4cuSIxfZ1dXX497//jT/84Q9o3bo14uPj0a1bNzz99NPIycmxeR6TyYRPPvkEQ4YMQcuWLZGWloYHHnjA5me70bZt2zB58mR07NgRsbGxSEtLw+jRo20ODy1fvhyRkZEYPnw4jEYjPv74YwwdOhQpKSmIjIzEhQsX7J6zuYnI5nlAc+bMQXV1NWbPno2bb74Z8fHxaNOmDR5++GGcPXvW6nGHDx+OyMhILF++HHl5eZg2bRoyMzMRFxeHrl274m9/+xt0Op3Vfc312Kq/8ec2M3+dmHXr1q3hOObPYE9kZCTmzp0LAFixYkWT/a21z759+/Dwww+jY8eOiIuLQ3p6OkaPHo3Vq1fbPZc1q1evRkJCAqKjo7FkyZIm7+Xk5OC5557DTTfdhJYtWyIpKQm33norFi5ciNraWotj3fjvumvXLowdOxZpaWlISEhA//79sWjRIphMJqu1HDp0CFOnTkXnzp0RFxeHVq1aoUuXLrj//vsxf/58m/sReQsOyZFs1dbWYurUqVizZg0AICYmBh07dkRubi5+/PFH/PTTT/i///s/PPXUU1b3r6+vx5gxY7B582akp6ejTZs2OH36NNavX49t27bhhx9+QM+ePS32e+utt/DPf/4TJpMJISEhyMjIgMFgwIULF5CdnY3y8nJ88MEHDduXl5dj7Nix2LVrFwAgNTUVkZGROHXqFD755BN88cUXWLJkCe666y6Lcz3++ONYuXIlgOtDXfHx8dizZw9GjhyJV1991WbbmEwmzJw5E4sWLQJw/cLdsWNHXLlyBZs3b8bmzZsxdepUvPnmmzb3nzx5MtauXYukpCS0bdvWobDkqPLyctx+++34/fff0a5dO6Snp+P06dP49ttvsXXrVvzyyy9ISUmxuu+FCxcwZMgQ6HQ6dOzYERERETh58iTef/99rFu3Dj/88APi4+NdrjEpKQl9+/bF7t27AQA9evRAcHBwk/ft6du3L/Ly8pCXl4fY2Fi0adPG5rYLFizA3/72N5hMJkRGRqJTp05N/r3Gjx+PhQsXIiDAsd9jFy9ejJkzZ0KlUuGTTz5pMuH8iy++wFNPPYXa2lqEhoYiLS0NVVVVOHz4MH777Td89913+PrrrxEeHm712MuXL8dTTz0FrVaL1NRU5ObmIjs7G88//zwuXryIWbNmNdn+559/xoQJE1BXVweNRoO2bdtCqVQiPz8fmzZtwqZNm/D4449DqeQlh7wXe5hItl588UWsWbMGHTt2xLp163DmzBls27YN586dw6JFixAaGop//OMf2L59u9X9v/32W5w5cwa//PILDh48iO3btyM7Oxt9+vRBdXU1/va3v1nss3z5csyaNQsKhQIvvvgicnJy8Ouvv2LXrl3Iy8vDd999hyFDhjTZZ+bMmdi1axdiYmLw008/4dChQ/jll19w4sQJPPDAA6iursbUqVMtAsmyZcuwcuVKKJVKfPTRR8jOzsaWLVtw6tQpTJw4sdnA9N5772HRokVo1aoVVq5cifPnz2Pbtm04deoUvv76a8TGxmLx4sUNYexGe/bswa+//opvvvkGx44dw+bNm3Hy5Em0atXK3j+LQxYvXozAwEAcOHAAe/bswa5du7B//35kZGSgpKQEs2fPtrnvO++8g9TUVBw+fBjbt2/H7t27sXPnTqSlpeHMmTM2A7KzJk2ahHXr1jX8fenSpVi3bl3Dn0mTJtk9xrp16zBx4kQAwG233dZk/8bH3rZtW0NYev7553H69Gls2bIFx48fx+LFi6FSqbBy5UosWLDAodpfe+01zJgxAxEREfjuu++ahKXdu3fjiSeegMlkwpw5c3DhwgXs3LkThw4dwt69e9GzZ0/s3bsXL7zwgs3jT58+HbNmzcKZM2ewZcsWnDlzBv/4xz8AXA9+586da7L9K6+8grq6Ojz99NM4ffo0du7ciW3btuHMmTM4evQoXn31VYeDIJFc8SuYZOn06dP4z3/+g4iICKxatQp9+/Zt8v7YsWPx4osvwmQy4d1337V6jLq6Onz44Yfo3r17w2vR0dENQyi7du1qcmeUXq/Ha6+9BgD4y1/+gueffx5hYWEN7ysUCtxyyy0YP358w2sXLlxoCCVvvfUW+vXr1/BeREQEPvzwQ7Ru3RoVFRV4//33G94zmUx45513AABTp07FuHHjGt4LDQ3FvHnzkJqaavVz6XQ6vPnmmwgMDMRnn32GP/zhD03eHzZsGN5++20AaDjHjerr6/Hmm29i6NChDa8plUrRegACAgKwdOlSpKenN7yWmpqKv//97wDQJEzcyGQy4T//+U+THp6OHTs29Opt2LDB7pCq3Lz11lswmUy444478OKLLyIoKKjhvTFjxuD//b//BwCYN2+e1eEyM4PBgMcffxxvv/02kpKSsG7dOovvjVdeeQUGgwEvv/wyHn/8cahUqob32rZti2XLlkGtVmPFihW4fPmy1fOMHTsWTzzxBAIDAxtemz59OjIzM2EymbB+/fom258+fbphm9DQ0CbvJScn4+mnn2ZgIq/Hr2CSpdWrV8NoNOK2226zOXQzcuRIANdv666vr7d4v1OnTujfv7/F6926dUNwcDBMJlOT35T37NmD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73+jW7du6NixIwwGA06cOAGTyYT09HTMnz/fYp9p06bhiy++wPnz5zFkyBC0adMGwcHBOHHiBBISEjBz5kyL1anNxo8fj7///e/461//iiVLliAmJgYKhQIPPvhgw6KUzRk6dCji4uKQl5eHTp06ISMjoyH0mB9RM3jwYLz22mv4+9//jrlz5+Kjjz5Ceno6rly5gvz8fADAuHHjbC47cKP27dvj+++/x8iRI/Htt9/CYDBgyZIlCAoKwsCBA7F48WI8+eSTeOuttzBv3jxkZGRAo9FAp9Ph/PnzTQKbq4xGI1avXo3Vq1dDpVIhPT0darUaV69excWLFwFcn/j92GOPiXZOIimwh4lkS6VSYdGiRfjuu+8wevRoaDQaZGdnIzs7G0qlEsOHD8f8+fNtXgiFmjFjBjZt2oSxY8ciOjoaJ06cQH5+Plq3bo3HHnvM4qIWHh6ONWvW4K233kKfPn1QVFSE33//HdHR0Q2hyNpjURQKBT766CO888476Nq1K4qLi5GTk4NevXphzZo1uOeee2zWqFAo8H//93/YvHkzJk6ciNjYWJw8eRJHjhyBwWDAsGHD8MYbb9gc1pOz1q1bY+vWrRg3bhyKiopw9uxZJCUl4YknnsDmzZvRsmVLi320Wi3Wr1+PyZMnIy4uDufPn4dOp8PDDz+Mbdu2Wd3HbNq0aXjttdfQuXNn5OXlYefOndixY0fDxd4etVqN1atXY+TIkQgJCcGhQ4ewY8cOizWOnnzySWzYsAGjRo1CSEgIjh49iurqatx666345JNP8NFHHzm1uGPbtm3x448/IikpCWvXrkVWVlZDD899992HvXv34umnn0aHDh2Ql5eHgwcPori4GD179sTMmTOxbds2h8/VnPDwcCxevBiTJk1C27ZtUVhYiEOHDqGsrAz9+vXDG2+8gR9//NHtd2YSuZtCp9PxiYhEJLnhw4djx44dWLBggUM9O0REnsQeJiIiIiI7GJiIiIiI7GBgIiIiIrKDgYmIiIjIDk76JiIiIrKDPUxEREREdjAwEREREdnBwERERERkBwMTERERkR0MTERERER2MDARERER2cHARERERGQHAxMRERGRHQxMRERERHb8f7b1L+I6fS2sAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 600x600 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "NewData['encoder input tokens']=NewData['encoder_inputs'].apply(lambda x:len(x.split()))\n",
    "NewData['decoder input tokens']=NewData['decoder_inputs'].apply(lambda x:len(x.split()))\n",
    "NewData['decoder target tokens']=NewData['decoder_targets'].apply(lambda x:len(x.split()))\n",
    "plt.style.use('fivethirtyeight')\n",
    "fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(20,5))\n",
    "sns.set_palette('Set2')\n",
    "sns.histplot(x=NewData['encoder input tokens'],data=NewData,kde=True,ax=ax[0])\n",
    "sns.histplot(x=NewData['decoder input tokens'],data=NewData,kde=True,ax=ax[1])\n",
    "sns.histplot(x=NewData['decoder target tokens'],data=NewData,kde=True,ax=ax[2])\n",
    "sns.jointplot(x='encoder input tokens',y='decoder target tokens',data=NewData,kind='kde',fill=True,cmap='YlGnBu')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "After preprocessing: anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen , such as shrimp in your case . it can affect multiple systems in your body and can be life threatening if not treated promptly . common symptoms include swelling of the lips , face , or throat , difficulty breathing , hives or rash , and a drop in blood pressure leading to dizziness or fainting .\n",
      "Max encoder input length: 77\n",
      "Max decoder input length: 133\n",
      "Max decoder target length: 132\n"
     ]
    }
   ],
   "source": [
    "max_encoder_input_row = NewData['encoder_inputs'].str.split().str.len().argmax()\n",
    "max_encoder_input = NewData.loc[max_encoder_input_row, 'encoder_inputs']\n",
    "\n",
    "print(f\"After preprocessing: {' '.join(max_encoder_input.split())}\")\n",
    "print(f\"Max encoder input length: {NewData['encoder_inputs'].str.split().str.len().max()}\")\n",
    "print(f\"Max decoder input length: {NewData['decoder_inputs'].str.split().str.len().max()}\")\n",
    "print(f\"Max decoder target length: {NewData['decoder_targets'].str.split().str.len().max()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Patient</th>\n",
       "      <th>Doctor</th>\n",
       "      <th>encoder_inputs</th>\n",
       "      <th>decoder_targets</th>\n",
       "      <th>decoder_inputs</th>\n",
       "      <th>encoder input tokens</th>\n",
       "      <th>decoder input tokens</th>\n",
       "      <th>decoder target tokens</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Doctor, I've been feeling really strange lately. I had a reaction after eating something, and my throat felt like it was closing up. I'm really worried. Can you help me?</td>\n",
       "      <td>Of course, I'll do my best to help you. It sounds like you may have experienced an allergic reaction. Can you tell me more about what happened?</td>\n",
       "      <td>doctor ,  i ' ve been feeling really strange lately .  i had a reaction after eating something ,  and my throat felt like it was closing up .  i ' m really worried .  can you help me ?</td>\n",
       "      <td>of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  &lt;end&gt;</td>\n",
       "      <td>40</td>\n",
       "      <td>35</td>\n",
       "      <td>34</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Well, I was at a restaurant and I had this dish with shrimp. Shortly after I finished eating, my lips started to swell, and I had difficulty breathing. I've never had this happen before, and it scared me.</td>\n",
       "      <td>I understand your concern. Based on your symptoms, it's possible that you had a severe allergic reaction called anaphylaxis. This is a serious condition that requires immediate medical attention. Have you experienced any other symptoms, such as hives, itching, or lightheadedness?</td>\n",
       "      <td>well ,  i was at a restaurant and i had this dish with shrimp .  shortly after i finished eating ,  my lips started to swell ,  and i had difficulty breathing .  i ' ve never had this happen before ,  and it scared me .</td>\n",
       "      <td>i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  &lt;end&gt;</td>\n",
       "      <td>47</td>\n",
       "      <td>53</td>\n",
       "      <td>52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Yes, actually, I did notice some hives on my arms and chest, and I felt lightheaded and dizzy. I had no idea this could happen from eating shrimp. What exactly is anaphylaxis?</td>\n",
       "      <td>That's probably what's causing your reaction. Pork is a common allergen, and it can cause a variety of symptoms, including hives, difficulty breathing, and even anaphylaxis.</td>\n",
       "      <td>yes ,  actually ,  i did notice some hives on my arms and chest ,  and i felt lightheaded and dizzy .  i had no idea this could happen from eating shrimp .  what exactly is anaphylaxis ?</td>\n",
       "      <td>that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  &lt;end&gt;</td>\n",
       "      <td>38</td>\n",
       "      <td>38</td>\n",
       "      <td>37</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen, such as shrimp in your case. It can affect multiple systems in your body and can be life-threatening if not treated promptly. Common symptoms include swelling of the lips, face, or throat, difficulty breathing, hives or rash, and a drop in blood pressure leading to dizziness or fainting.</td>\n",
       "      <td>That sounds really serious! I had no idea an allergic reaction could be so dangerous. What should I do if it happens again?</td>\n",
       "      <td>anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen ,  such as shrimp in your case .  it can affect multiple systems in your body and can be life threatening if not treated promptly .  common symptoms include swelling of the lips ,  face ,  or throat ,  difficulty breathing ,  hives or rash ,  and a drop in blood pressure leading to dizziness or fainting .</td>\n",
       "      <td>that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  &lt;end&gt;</td>\n",
       "      <td>77</td>\n",
       "      <td>28</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Good morning, Doctor. I've been experiencing some unusual symptoms, and I'm not sure what's going on. I noticed some skin issues and recently had a blood test done. Can you please take a look at the reports and help me understand what's happening?</td>\n",
       "      <td>Good morning. Of course, I'll be happy to assist you. Please hand me your reports, and let's discuss your symptoms in detail. What specific skin issues have you been experiencing?</td>\n",
       "      <td>good morning ,  doctor .  i ' ve been experiencing some unusual symptoms ,  and i ' m not sure what ' s going on .  i noticed some skin issues and recently had a blood test done .  can you please take a look at the reports and help me understand what ' s happening ?</td>\n",
       "      <td>good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  &lt;end&gt;</td>\n",
       "      <td>57</td>\n",
       "      <td>42</td>\n",
       "      <td>41</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                                                                                                                                                                                                                                                                                                                                                                   Patient  \\\n",
       "0                                                                                                                                                                                                                                                Doctor, I've been feeling really strange lately. I had a reaction after eating something, and my throat felt like it was closing up. I'm really worried. Can you help me?   \n",
       "1                                                                                                                                                                                                             Well, I was at a restaurant and I had this dish with shrimp. Shortly after I finished eating, my lips started to swell, and I had difficulty breathing. I've never had this happen before, and it scared me.   \n",
       "2                                                                                                                                                                                                                                          Yes, actually, I did notice some hives on my arms and chest, and I felt lightheaded and dizzy. I had no idea this could happen from eating shrimp. What exactly is anaphylaxis?   \n",
       "3  Anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen, such as shrimp in your case. It can affect multiple systems in your body and can be life-threatening if not treated promptly. Common symptoms include swelling of the lips, face, or throat, difficulty breathing, hives or rash, and a drop in blood pressure leading to dizziness or fainting.   \n",
       "4                                                                                                                                                                  Good morning, Doctor. I've been experiencing some unusual symptoms, and I'm not sure what's going on. I noticed some skin issues and recently had a blood test done. Can you please take a look at the reports and help me understand what's happening?   \n",
       "\n",
       "                                                                                                                                                                                                                                                                                     Doctor  \\\n",
       "0                                                                                                                                           Of course, I'll do my best to help you. It sounds like you may have experienced an allergic reaction. Can you tell me more about what happened?   \n",
       "1  I understand your concern. Based on your symptoms, it's possible that you had a severe allergic reaction called anaphylaxis. This is a serious condition that requires immediate medical attention. Have you experienced any other symptoms, such as hives, itching, or lightheadedness?   \n",
       "2                                                                                                             That's probably what's causing your reaction. Pork is a common allergen, and it can cause a variety of symptoms, including hives, difficulty breathing, and even anaphylaxis.   \n",
       "3                                                                                                                                                               That sounds really serious! I had no idea an allergic reaction could be so dangerous. What should I do if it happens again?   \n",
       "4                                                                                                       Good morning. Of course, I'll be happy to assist you. Please hand me your reports, and let's discuss your symptoms in detail. What specific skin issues have you been experiencing?   \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                                                                                                                                              encoder_inputs  \\\n",
       "0                                                                                                                                                                                                                                                  doctor ,  i ' ve been feeling really strange lately .  i had a reaction after eating something ,  and my throat felt like it was closing up .  i ' m really worried .  can you help me ?    \n",
       "1                                                                                                                                                                                                               well ,  i was at a restaurant and i had this dish with shrimp .  shortly after i finished eating ,  my lips started to swell ,  and i had difficulty breathing .  i ' ve never had this happen before ,  and it scared me .    \n",
       "2                                                                                                                                                                                                                                                yes ,  actually ,  i did notice some hives on my arms and chest ,  and i felt lightheaded and dizzy .  i had no idea this could happen from eating shrimp .  what exactly is anaphylaxis ?    \n",
       "3  anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen ,  such as shrimp in your case .  it can affect multiple systems in your body and can be life threatening if not treated promptly .  common symptoms include swelling of the lips ,  face ,  or throat ,  difficulty breathing ,  hives or rash ,  and a drop in blood pressure leading to dizziness or fainting .    \n",
       "4                                                                                                                                                                good morning ,  doctor .  i ' ve been experiencing some unusual symptoms ,  and i ' m not sure what ' s going on .  i noticed some skin issues and recently had a blood test done .  can you please take a look at the reports and help me understand what ' s happening ?    \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                    decoder_targets  \\\n",
       "0                                                                                                                                                   of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  <end>   \n",
       "1  i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  <end>   \n",
       "2                                                                                                               that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  <end>   \n",
       "3                                                                                                                                                                           that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  <end>   \n",
       "4                                                                                                         good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  <end>   \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                             decoder_inputs  \\\n",
       "0                                                                                                                                                   <start> of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  <end>   \n",
       "1  <start> i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  <end>   \n",
       "2                                                                                                               <start> that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  <end>   \n",
       "3                                                                                                                                                                           <start> that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  <end>   \n",
       "4                                                                                                         <start> good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  <end>   \n",
       "\n",
       "   encoder input tokens  decoder input tokens  decoder target tokens  \n",
       "0                    40                    35                     34  \n",
       "1                    47                    53                     52  \n",
       "2                    38                    38                     37  \n",
       "3                    77                    28                     27  \n",
       "4                    57                    42                     41  "
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NewData.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "NewData.drop(columns=['Patient','Doctor','encoder input tokens','decoder input tokens','decoder target tokens'],axis=1,inplace=True)\n",
    "params={\n",
    "    \"vocab_size\":2500,\n",
    "    \"max_sequence_length\":40,\n",
    "    \"learning_rate\":0.008,\n",
    "    \"batch_size\":149,\n",
    "    \"lstm_cells\":256,\n",
    "    \"embedding_dim\":256,\n",
    "    \"buffer_size\":10000\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>encoder_inputs</th>\n",
       "      <th>decoder_targets</th>\n",
       "      <th>decoder_inputs</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>doctor ,  i ' ve been feeling really strange lately .  i had a reaction after eating something ,  and my throat felt like it was closing up .  i ' m really worried .  can you help me ?</td>\n",
       "      <td>of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>well ,  i was at a restaurant and i had this dish with shrimp .  shortly after i finished eating ,  my lips started to swell ,  and i had difficulty breathing .  i ' ve never had this happen before ,  and it scared me .</td>\n",
       "      <td>i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>yes ,  actually ,  i did notice some hives on my arms and chest ,  and i felt lightheaded and dizzy .  i had no idea this could happen from eating shrimp .  what exactly is anaphylaxis ?</td>\n",
       "      <td>that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen ,  such as shrimp in your case .  it can affect multiple systems in your body and can be life threatening if not treated promptly .  common symptoms include swelling of the lips ,  face ,  or throat ,  difficulty breathing ,  hives or rash ,  and a drop in blood pressure leading to dizziness or fainting .</td>\n",
       "      <td>that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>good morning ,  doctor .  i ' ve been experiencing some unusual symptoms ,  and i ' m not sure what ' s going on .  i noticed some skin issues and recently had a blood test done .  can you please take a look at the reports and help me understand what ' s happening ?</td>\n",
       "      <td>good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>well ,  i ' ve been having frequent rashes and hives on different parts of my body .  they appear as red ,  itchy patches ,  and they come and go randomly .  it ' s quite uncomfortable ,  and i ' m not sure what triggers them .</td>\n",
       "      <td>i see .  skin rashes and hives can be indicative of an allergic reaction .  it ' s important to identify the underlying cause .  now ,  let ' s take a look at your blood test results .  could you please pass them to me ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt;  i see .  skin rashes and hives can be indicative of an allergic reaction .  it ' s important to identify the underlying cause .  now ,  let ' s take a look at your blood test results .  could you please pass them to me ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>here are the reports ,  doctor .  i hope they can provide some insight into my condition .</td>\n",
       "      <td>thank you .  let me review these reports .   &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; thank you .  let me review these reports .   &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>doctor ,  i had a really scary experience after eating pork yesterday .  my face swelled up ,  and i had difficulty breathing .  i think it might have been an allergic reaction .  can you help me understand what happened ?</td>\n",
       "      <td>i ' m sorry to hear about your distressing experience .  allergic reactions can indeed occur after consuming certain foods .  let ' s discuss your symptoms in more detail .  did you notice any other reactions apart from the facial swelling and difficulty breathing ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt;  i ' m sorry to hear about your distressing experience .  allergic reactions can indeed occur after consuming certain foods .  let ' s discuss your symptoms in more detail .  did you notice any other reactions apart from the facial swelling and difficulty breathing ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>yes ,  i also had hives on my body ,  and i felt quite dizzy .  it was really frightening ,  and i had no idea that this could happen from eating pork .  is it possible that i have an allergy to pork ?</td>\n",
       "      <td>it ' s possible that you have developed an allergy to pork .  allergic reactions can vary from person to person ,  and some individuals can be allergic to specific types of meat .  to better understand your condition ,  it would be helpful to review your skin and blood test reports .  can you please provide me with those ?  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt;  it ' s possible that you have developed an allergy to pork .  allergic reactions can vary from person to person ,  and some individuals can be allergic to specific types of meat .  to better understand your condition ,  it would be helpful to review your skin and blood test reports .  can you please provide me with those ?  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>certainly ,  doctor .  here are the reports .  i hope they can shed some light on what ' s happening .</td>\n",
       "      <td>thank you .  let me take a look at the report  . based on your blood test results ,  your ige levels are elevated ,  indicating a possible allergic reaction .  additionally ,  your skin prick test shows a positive reaction to pork allergens ,  further suggesting an allergy to pork .  &lt;end&gt;</td>\n",
       "      <td>&lt;start&gt; thank you .  let me take a look at the report  . based on your blood test results ,  your ige levels are elevated ,  indicating a possible allergic reaction .  additionally ,  your skin prick test shows a positive reaction to pork allergens ,  further suggesting an allergy to pork .  &lt;end&gt;</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                                                                                                                                                                                                                                                                                                                                                                              encoder_inputs  \\\n",
       "0                                                                                                                                                                                                                                                  doctor ,  i ' ve been feeling really strange lately .  i had a reaction after eating something ,  and my throat felt like it was closing up .  i ' m really worried .  can you help me ?    \n",
       "1                                                                                                                                                                                                               well ,  i was at a restaurant and i had this dish with shrimp .  shortly after i finished eating ,  my lips started to swell ,  and i had difficulty breathing .  i ' ve never had this happen before ,  and it scared me .    \n",
       "2                                                                                                                                                                                                                                                yes ,  actually ,  i did notice some hives on my arms and chest ,  and i felt lightheaded and dizzy .  i had no idea this could happen from eating shrimp .  what exactly is anaphylaxis ?    \n",
       "3  anaphylaxis is a severe allergic reaction that can occur within minutes or even seconds after exposure to an allergen ,  such as shrimp in your case .  it can affect multiple systems in your body and can be life threatening if not treated promptly .  common symptoms include swelling of the lips ,  face ,  or throat ,  difficulty breathing ,  hives or rash ,  and a drop in blood pressure leading to dizziness or fainting .    \n",
       "4                                                                                                                                                                good morning ,  doctor .  i ' ve been experiencing some unusual symptoms ,  and i ' m not sure what ' s going on .  i noticed some skin issues and recently had a blood test done .  can you please take a look at the reports and help me understand what ' s happening ?    \n",
       "5                                                                                                                                                                                                       well ,  i ' ve been having frequent rashes and hives on different parts of my body .  they appear as red ,  itchy patches ,  and they come and go randomly .  it ' s quite uncomfortable ,  and i ' m not sure what triggers them .    \n",
       "6                                                                                                                                                                                                                                                                                                                                                here are the reports ,  doctor .  i hope they can provide some insight into my condition .    \n",
       "7                                                                                                                                                                                                            doctor ,  i had a really scary experience after eating pork yesterday .  my face swelled up ,  and i had difficulty breathing .  i think it might have been an allergic reaction .  can you help me understand what happened ?    \n",
       "8                                                                                                                                                                                                                                 yes ,  i also had hives on my body ,  and i felt quite dizzy .  it was really frightening ,  and i had no idea that this could happen from eating pork .  is it possible that i have an allergy to pork ?    \n",
       "9                                                                                                                                                                                                                                                                                                                                    certainly ,  doctor .  here are the reports .  i hope they can shed some light on what ' s happening .    \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                                                decoder_targets  \\\n",
       "0                                                                                                                                                                               of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  <end>   \n",
       "1                              i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  <end>   \n",
       "2                                                                                                                                           that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  <end>   \n",
       "3                                                                                                                                                                                                       that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  <end>   \n",
       "4                                                                                                                                     good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  <end>   \n",
       "5                                                                                                           i see .  skin rashes and hives can be indicative of an allergic reaction .  it ' s important to identify the underlying cause .  now ,  let ' s take a look at your blood test results .  could you please pass them to me ?  <end>   \n",
       "6                                                                                                                                                                                                                                                                                            thank you .  let me review these reports .   <end>   \n",
       "7                                                             i ' m sorry to hear about your distressing experience .  allergic reactions can indeed occur after consuming certain foods .  let ' s discuss your symptoms in more detail .  did you notice any other reactions apart from the facial swelling and difficulty breathing ?  <end>   \n",
       "8   it ' s possible that you have developed an allergy to pork .  allergic reactions can vary from person to person ,  and some individuals can be allergic to specific types of meat .  to better understand your condition ,  it would be helpful to review your skin and blood test reports .  can you please provide me with those ?  <end>   \n",
       "9                                            thank you .  let me take a look at the report  . based on your blood test results ,  your ige levels are elevated ,  indicating a possible allergic reaction .  additionally ,  your skin prick test shows a positive reaction to pork allergens ,  further suggesting an allergy to pork .  <end>   \n",
       "\n",
       "                                                                                                                                                                                                                                                                                                                                         decoder_inputs  \n",
       "0                                                                                                                                                                               <start> of course ,  i ' ll do my best to help you .  it sounds like you may have experienced an allergic reaction .  can you tell me more about what happened ?  <end>  \n",
       "1                              <start> i understand your concern .  based on your symptoms ,  it ' s possible that you had a severe allergic reaction called anaphylaxis .  this is a serious condition that requires immediate medical attention .  have you experienced any other symptoms ,  such as hives ,  itching ,  or lightheadedness ?  <end>  \n",
       "2                                                                                                                                           <start> that ' s probably what ' s causing your reaction .  pork is a common allergen ,  and it can cause a variety of symptoms ,  including hives ,  difficulty breathing ,  and even anaphylaxis .  <end>  \n",
       "3                                                                                                                                                                                                       <start> that sounds really serious !  i had no idea an allergic reaction could be so dangerous .  what should i do if it happens again ?  <end>  \n",
       "4                                                                                                                                     <start> good morning .  of course ,  i ' ll be happy to assist you .  please hand me your reports ,  and let ' s discuss your symptoms in detail .  what specific skin issues have you been experiencing ?  <end>  \n",
       "5                                                                                                          <start>  i see .  skin rashes and hives can be indicative of an allergic reaction .  it ' s important to identify the underlying cause .  now ,  let ' s take a look at your blood test results .  could you please pass them to me ?  <end>  \n",
       "6                                                                                                                                                                                                                                                                                            <start> thank you .  let me review these reports .   <end>  \n",
       "7                                                            <start>  i ' m sorry to hear about your distressing experience .  allergic reactions can indeed occur after consuming certain foods .  let ' s discuss your symptoms in more detail .  did you notice any other reactions apart from the facial swelling and difficulty breathing ?  <end>  \n",
       "8  <start>  it ' s possible that you have developed an allergy to pork .  allergic reactions can vary from person to person ,  and some individuals can be allergic to specific types of meat .  to better understand your condition ,  it would be helpful to review your skin and blood test reports .  can you please provide me with those ?  <end>  \n",
       "9                                            <start> thank you .  let me take a look at the report  . based on your blood test results ,  your ige levels are elevated ,  indicating a possible allergic reaction .  additionally ,  your skin prick test shows a positive reaction to pork allergens ,  further suggesting an allergy to pork .  <end>  "
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "learning_rate=params['learning_rate']\n",
    "batch_size=params['batch_size']\n",
    "embedding_dim=params['embedding_dim']\n",
    "lstm_cells=params['lstm_cells']\n",
    "vocab_size=params['vocab_size']\n",
    "buffer_size=params['buffer_size']\n",
    "max_sequence_length=params['max_sequence_length']\n",
    "NewData.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Vocab size: 1016\n",
      "['', '[UNK]', '.', ',', 'i', \"'\", '<end>', 'to', 'you', 'and', 'the', 'it']\n"
     ]
    }
   ],
   "source": [
    "vectorize_layer = TextVectorization(\n",
    "    max_tokens=vocab_size,\n",
    "    standardize=None,\n",
    "    output_mode='int',\n",
    "    output_sequence_length=max_sequence_length\n",
    ")\n",
    "vectorize_layer.adapt(NewData['encoder_inputs'] + ' ' + NewData['decoder_targets'] + ' <start> <end>')\n",
    "vocab_size = len(vectorize_layer.get_vocabulary())\n",
    "print(f'Vocab size: {vocab_size}')\n",
    "print(f'{vectorize_layer.get_vocabulary()[:12]}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Thank you. Let me examine your skin and review the reports.\n",
      "Question to tokens: [  1   1   1  35  88  16  34   9 201  10]\n",
      "Encoder input shape: (268, 40)\n",
      "Decoder input shape: (268, 40)\n",
      "Decoder target shape: (268, 40)\n"
     ]
    }
   ],
   "source": [
    "def sequences2ids(sequence):\n",
    "    return vectorize_layer(sequence)\n",
    "\n",
    "def ids2sequences(ids):\n",
    "    decode=''\n",
    "    if type(ids)==int:\n",
    "        ids=[ids]\n",
    "    for id in ids:\n",
    "        decode+=vectorize_layer.get_vocabulary()[id]+' '\n",
    "    return decode\n",
    "\n",
    "x = sequences2ids(NewData['encoder_inputs'])\n",
    "yd = sequences2ids(NewData['decoder_inputs'])\n",
    "y = sequences2ids(NewData['decoder_targets'])\n",
    "\n",
    "print(f'Thank you. Let me examine your skin and review the reports.')\n",
    "print(f'Question to tokens: {sequences2ids(\"Thank you. Let me examine your skin and review the reports.\")[:10]}')\n",
    "print(f'Encoder input shape: {x.shape}')\n",
    "print(f'Decoder input shape: {yd.shape}')\n",
    "print(f'Decoder target shape: {y.shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Encoder input: [ 33   3   4   5  41  62  93  64 482 313   2   4] ...\n",
      "Decoder input: [ 13  23 258   3   4   5  50 106  24 180   7  52] ...\n",
      "Decoder target: [ 23 258   3   4   5  50 106  24 180   7  52   8] ...\n"
     ]
    }
   ],
   "source": [
    "print(f'Encoder input: {x[0][:12]} ...')\n",
    "print(f'Decoder input: {yd[0][:12]} ...')    # shifted by one time step of the target as input to decoder is the output of the previous timestep\n",
    "print(f'Decoder target: {y[0][:12]} ...')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of train batches: 2\n",
      "Number of training data: 298\n",
      "Number of validation batches: 1\n",
      "Number of validation data: 149\n",
      "Encoder Input shape (with batches): (149, 40)\n",
      "Decoder Input shape (with batches): (149, 40)\n",
      "Target Output shape (with batches): (149, 40)\n"
     ]
    }
   ],
   "source": [
    "data=tf.data.Dataset.from_tensor_slices((x,yd,y))\n",
    "data=data.shuffle(buffer_size)\n",
    "\n",
    "train_data=data.take(int(.9*len(data)))\n",
    "train_data=train_data.cache()\n",
    "train_data=train_data.shuffle(buffer_size)\n",
    "train_data=train_data.batch(batch_size)\n",
    "train_data=train_data.prefetch(tf.data.AUTOTUNE)\n",
    "train_data_iterator=train_data.as_numpy_iterator()\n",
    "\n",
    "val_data=data.skip(int(.9*len(data))).take(int(.1*len(data)))\n",
    "val_data=val_data.batch(batch_size)\n",
    "val_data=val_data.prefetch(tf.data.AUTOTUNE)\n",
    "\n",
    "_=train_data_iterator.next()\n",
    "print(f'Number of train batches: {len(train_data)}')\n",
    "print(f'Number of training data: {len(train_data)*batch_size}')\n",
    "print(f'Number of validation batches: {len(val_data)}')\n",
    "print(f'Number of validation data: {len(val_data)*batch_size}')\n",
    "print(f'Encoder Input shape (with batches): {_[0].shape}')\n",
    "print(f'Decoder Input shape (with batches): {_[1].shape}')\n",
    "print(f'Target Output shape (with batches): {_[2].shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(<tf.Tensor: shape=(149, 256), dtype=float32, numpy=\n",
       " array([[ 0.09998038,  0.26333702, -0.06089902, ..., -0.21232115,\n",
       "          0.12410013,  0.27586135],\n",
       "        [-0.09345426,  0.10809281,  0.13994898, ...,  0.04045346,\n",
       "          0.04221269,  0.07851808],\n",
       "        [ 0.02720955,  0.05634911, -0.2269569 , ..., -0.01957801,\n",
       "          0.32200328, -0.08247767],\n",
       "        ...,\n",
       "        [ 0.07775754, -0.01114887, -0.15913355, ..., -0.06104032,\n",
       "          0.0525138 , -0.20916015],\n",
       "        [-0.17113934, -0.07912529,  0.01647902, ..., -0.02332988,\n",
       "          0.09387417, -0.10062128],\n",
       "        [ 0.00247808,  0.0785782 ,  0.16000254, ..., -0.08455516,\n",
       "         -0.0671699 ,  0.03163897]], dtype=float32)>,\n",
       " <tf.Tensor: shape=(149, 256), dtype=float32, numpy=\n",
       " array([[ 0.19015819,  0.6053947 , -0.10991258, ..., -0.35669845,\n",
       "          0.2377929 ,  0.6072244 ],\n",
       "        [-0.15494579,  0.17510627,  0.30047044, ...,  0.08151735,\n",
       "          0.07957618,  0.13905375],\n",
       "        [ 0.05646555,  0.12155945, -0.4664234 , ..., -0.02659739,\n",
       "          0.5762466 , -0.17631185],\n",
       "        ...,\n",
       "        [ 0.146511  , -0.02373511, -0.48195994, ..., -0.10673541,\n",
       "          0.10361052, -0.5950215 ],\n",
       "        [-0.4177329 , -0.11200631,  0.02890308, ..., -0.05279912,\n",
       "          0.23298159, -0.3156003 ],\n",
       "        [ 0.00500037,  0.16280754,  0.28801394, ..., -0.15189336,\n",
       "         -0.11520901,  0.0655641 ]], dtype=float32)>)"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class Encoder(tf.keras.models.Model):\n",
    "    def __init__(self,units,embedding_dim,vocab_size,*args,**kwargs) -> None:\n",
    "        super().__init__(*args,**kwargs)\n",
    "        self.units=units\n",
    "        self.vocab_size=vocab_size\n",
    "        self.embedding_dim=embedding_dim\n",
    "        self.embedding=Embedding(\n",
    "            vocab_size,\n",
    "            embedding_dim,\n",
    "            name='encoder_embedding',\n",
    "            mask_zero=True,\n",
    "            embeddings_initializer=tf.keras.initializers.GlorotNormal()\n",
    "        )\n",
    "        self.normalize=LayerNormalization()\n",
    "        self.lstm=LSTM(\n",
    "            units,\n",
    "            dropout=.4,\n",
    "            return_state=True,\n",
    "            return_sequences=True,\n",
    "            name='encoder_lstm',\n",
    "            kernel_initializer=tf.keras.initializers.GlorotNormal()\n",
    "        )\n",
    "    \n",
    "    def call(self,encoder_inputs):\n",
    "        self.inputs=encoder_inputs\n",
    "        x=self.embedding(encoder_inputs)\n",
    "        x=self.normalize(x)\n",
    "        x=Dropout(.4)(x)\n",
    "        encoder_outputs,encoder_state_h,encoder_state_c=self.lstm(x)\n",
    "        self.outputs=[encoder_state_h,encoder_state_c]\n",
    "        return encoder_state_h,encoder_state_c\n",
    "\n",
    "encoder=Encoder(lstm_cells,embedding_dim,vocab_size,name='encoder')\n",
    "encoder.call(_[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(1, 40, 1016), dtype=float32, numpy=\n",
       "array([[[3.5283252e-04, 3.6954320e-05, 6.7612465e-04, ...,\n",
       "         5.2436271e-06, 1.3179190e-03, 2.0732540e-04],\n",
       "        [1.3289067e-04, 1.2998210e-04, 9.3526929e-04, ...,\n",
       "         2.6049852e-04, 1.3381122e-03, 2.4023277e-03],\n",
       "        [1.0197279e-03, 1.5900445e-05, 5.2474428e-04, ...,\n",
       "         7.9252568e-05, 3.7345334e-04, 1.1599150e-03],\n",
       "        ...,\n",
       "        [3.6178302e-04, 6.8119625e-05, 4.6956484e-04, ...,\n",
       "         1.2594953e-04, 1.4632441e-03, 3.1318030e-04],\n",
       "        [5.9932259e-05, 7.2296866e-04, 3.5353794e-05, ...,\n",
       "         7.1153576e-05, 4.6500629e-03, 2.4220526e-04],\n",
       "        [2.6750925e-04, 3.0926219e-03, 8.1349531e-04, ...,\n",
       "         1.3350867e-04, 6.6905319e-05, 2.4227297e-03]]], dtype=float32)>"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class Decoder(tf.keras.models.Model):\n",
    "    def __init__(self,units,embedding_dim,vocab_size,*args,**kwargs) -> None:\n",
    "        super().__init__(*args,**kwargs)\n",
    "        self.units=units\n",
    "        self.embedding_dim=embedding_dim\n",
    "        self.vocab_size=vocab_size\n",
    "        self.embedding=Embedding(\n",
    "            vocab_size,\n",
    "            embedding_dim,\n",
    "            name='decoder_embedding',\n",
    "            mask_zero=True,\n",
    "            embeddings_initializer=tf.keras.initializers.HeNormal()\n",
    "        )\n",
    "        self.normalize1=LayerNormalization()  # Added new normalization layer here\n",
    "        self.lstm=LSTM(\n",
    "            units,\n",
    "            dropout=.4,\n",
    "            return_state=True,\n",
    "            return_sequences=True,\n",
    "            name='decoder_lstm',\n",
    "            kernel_initializer=tf.keras.initializers.HeNormal()\n",
    "        )\n",
    "        self.normalize2=LayerNormalization()  # Renamed old normalization layer\n",
    "        self.fc=Dense(\n",
    "            vocab_size,\n",
    "            activation='softmax',\n",
    "            name='decoder_dense',\n",
    "            kernel_initializer=tf.keras.initializers.HeNormal()\n",
    "        )\n",
    "    \n",
    "    def call(self,decoder_inputs,encoder_states):\n",
    "        x=self.embedding(decoder_inputs)\n",
    "        x=self.normalize1(x)  # Applied new normalization layer here\n",
    "        x=Dropout(.4)(x)\n",
    "        x,decoder_state_h,decoder_state_c=self.lstm(x,initial_state=encoder_states)\n",
    "        x=self.normalize2(x)  # Applied renamed normalization layer here\n",
    "        x=Dropout(.4)(x)\n",
    "        return self.fc(x)\n",
    "\n",
    "decoder=Decoder(lstm_cells,embedding_dim,vocab_size,name='decoder')\n",
    "decoder(_[1][:1],encoder(_[0][:1]))  # corrected parenthesis\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ChatBotTrainer(tf.keras.models.Model):\n",
    "    def __init__(self,encoder,decoder,*args,**kwargs):\n",
    "        super().__init__(*args,**kwargs)\n",
    "        self.encoder=encoder\n",
    "        self.decoder=decoder\n",
    "\n",
    "    def loss_fn(self,y_true,y_pred):\n",
    "        loss=self.loss(y_true,y_pred)\n",
    "        mask=tf.math.logical_not(tf.math.equal(y_true,0))\n",
    "        mask=tf.cast(mask,dtype=loss.dtype)\n",
    "        loss*=mask\n",
    "        return tf.reduce_mean(loss)\n",
    "    \n",
    "    def accuracy_fn(self,y_true,y_pred):\n",
    "        pred_values = tf.cast(tf.argmax(y_pred, axis=-1), dtype='int64')\n",
    "        correct = tf.cast(tf.equal(y_true, pred_values), dtype='float64')\n",
    "        mask = tf.cast(tf.greater(y_true, 0), dtype='float64')\n",
    "        n_correct = tf.keras.backend.sum(mask * correct)\n",
    "        n_total = tf.keras.backend.sum(mask)\n",
    "        return n_correct / n_total\n",
    "\n",
    "    def call(self,inputs):\n",
    "        encoder_inputs,decoder_inputs=inputs\n",
    "        encoder_states=self.encoder(encoder_inputs)\n",
    "        return self.decoder(decoder_inputs,encoder_states)\n",
    "\n",
    "    def train_step(self,batch):\n",
    "        encoder_inputs,decoder_inputs,y=batch\n",
    "        with tf.GradientTape() as tape:\n",
    "            encoder_states=self.encoder(encoder_inputs,training=True)\n",
    "            y_pred=self.decoder(decoder_inputs,encoder_states,training=True)\n",
    "            loss=self.loss_fn(y,y_pred)\n",
    "            acc=self.accuracy_fn(y,y_pred)\n",
    "\n",
    "        variables=self.encoder.trainable_variables+self.decoder.trainable_variables\n",
    "        grads=tape.gradient(loss,variables)\n",
    "        self.optimizer.apply_gradients(zip(grads,variables))\n",
    "        metrics={'loss':loss,'accuracy':acc}\n",
    "        return metrics\n",
    "    \n",
    "    def test_step(self,batch):\n",
    "        encoder_inputs,decoder_inputs,y=batch\n",
    "        encoder_states=self.encoder(encoder_inputs,training=True)\n",
    "        y_pred=self.decoder(decoder_inputs,encoder_states,training=True)\n",
    "        loss=self.loss_fn(y,y_pred)\n",
    "        acc=self.accuracy_fn(y,y_pred)\n",
    "        metrics={'loss':loss,'accuracy':acc}\n",
    "        return metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: shape=(149, 40, 1016), dtype=float32, numpy=\n",
       "array([[[3.52832663e-04, 3.69543195e-05, 6.76125521e-04, ...,\n",
       "         5.24362667e-06, 1.31791900e-03, 2.07325284e-04],\n",
       "        [1.32890666e-04, 1.29982160e-04, 9.35270218e-04, ...,\n",
       "         2.60498520e-04, 1.33811240e-03, 2.40232772e-03],\n",
       "        [1.01972942e-03, 1.59004449e-05, 5.24744508e-04, ...,\n",
       "         7.92526480e-05, 3.73453368e-04, 1.15991558e-03],\n",
       "        ...,\n",
       "        [3.61782819e-04, 6.81195888e-05, 4.69564344e-04, ...,\n",
       "         1.25949402e-04, 1.46324455e-03, 3.13180411e-04],\n",
       "        [5.99322193e-05, 7.22968543e-04, 3.53537907e-05, ...,\n",
       "         7.11535249e-05, 4.65006428e-03, 2.42205337e-04],\n",
       "        [2.67509342e-04, 3.09262471e-03, 8.13495193e-04, ...,\n",
       "         1.33508787e-04, 6.69052170e-05, 2.42273114e-03]],\n",
       "\n",
       "       [[4.81100491e-04, 6.88019427e-05, 3.69098125e-04, ...,\n",
       "         9.24661708e-06, 1.06741290e-03, 1.40251650e-03],\n",
       "        [1.29991607e-03, 2.12661806e-04, 2.48834054e-04, ...,\n",
       "         2.60041736e-04, 2.43285485e-03, 1.95990404e-04],\n",
       "        [2.30053658e-04, 2.82126479e-04, 1.98872251e-04, ...,\n",
       "         6.83828141e-04, 9.91830137e-04, 1.52593377e-04],\n",
       "        ...,\n",
       "        [9.01213090e-04, 6.58571735e-05, 1.74439665e-05, ...,\n",
       "         1.11394213e-04, 1.53575515e-04, 1.22364669e-04],\n",
       "        [2.21629045e-03, 1.35395187e-03, 1.33459180e-04, ...,\n",
       "         8.11644495e-05, 2.96842802e-04, 6.80401354e-05],\n",
       "        [2.77850567e-03, 1.61694061e-05, 1.50912338e-05, ...,\n",
       "         1.09134031e-04, 9.24160646e-04, 2.78031279e-04]],\n",
       "\n",
       "       [[1.76168061e-04, 5.02895236e-05, 2.12790852e-04, ...,\n",
       "         8.16454303e-06, 7.30671047e-04, 2.39761794e-04],\n",
       "        [1.61687945e-04, 5.09672609e-05, 2.41845948e-04, ...,\n",
       "         1.85295474e-04, 2.88989116e-03, 4.31575754e-04],\n",
       "        [2.73631816e-03, 1.52365783e-05, 1.42951286e-03, ...,\n",
       "         1.35726412e-04, 5.62512083e-04, 5.96704020e-04],\n",
       "        ...,\n",
       "        [2.02453317e-04, 1.26212335e-03, 2.45011179e-04, ...,\n",
       "         9.65088766e-05, 2.23096591e-04, 1.23949477e-03],\n",
       "        [2.02453317e-04, 1.26212335e-03, 2.45011179e-04, ...,\n",
       "         9.65088766e-05, 2.23096591e-04, 1.23949477e-03],\n",
       "        [2.02453317e-04, 1.26212335e-03, 2.45011179e-04, ...,\n",
       "         9.65088766e-05, 2.23096591e-04, 1.23949477e-03]],\n",
       "\n",
       "       ...,\n",
       "\n",
       "       [[4.76278045e-04, 4.74028457e-05, 2.68317846e-04, ...,\n",
       "         8.61224453e-06, 1.25814544e-03, 4.70889325e-04],\n",
       "        [1.65140000e-03, 1.75151785e-04, 1.28746455e-04, ...,\n",
       "         5.37347398e-04, 4.58872830e-03, 9.03698092e-05],\n",
       "        [5.48551150e-04, 3.21570784e-04, 5.97850420e-04, ...,\n",
       "         8.67592404e-04, 2.08239493e-04, 3.91443959e-04],\n",
       "        ...,\n",
       "        [7.65006189e-05, 1.32457382e-04, 8.92359589e-04, ...,\n",
       "         9.17971542e-04, 2.00080662e-03, 5.22786053e-04],\n",
       "        [2.58946908e-04, 1.01764395e-04, 8.63388559e-05, ...,\n",
       "         6.88683649e-04, 2.67062522e-03, 1.71325984e-04],\n",
       "        [2.76501378e-04, 5.65719383e-04, 3.00509448e-04, ...,\n",
       "         4.20969212e-04, 1.82869053e-03, 4.95118802e-05]],\n",
       "\n",
       "       [[4.64241864e-04, 3.41775667e-05, 6.90338726e-04, ...,\n",
       "         5.38236782e-06, 1.62685945e-04, 9.35639255e-04],\n",
       "        [9.19864455e-04, 1.73730994e-04, 1.42170378e-04, ...,\n",
       "         3.40574625e-04, 1.13624905e-03, 9.48035740e-05],\n",
       "        [5.21949958e-04, 5.09618490e-04, 2.34143998e-04, ...,\n",
       "         2.36679305e-04, 4.79675480e-04, 3.55530414e-04],\n",
       "        ...,\n",
       "        [1.25190840e-04, 3.13522585e-04, 4.89055878e-04, ...,\n",
       "         6.38576748e-05, 1.51719549e-04, 1.83751259e-03],\n",
       "        [1.25190840e-04, 3.13522585e-04, 4.89055878e-04, ...,\n",
       "         6.38576748e-05, 1.51719549e-04, 1.83751259e-03],\n",
       "        [1.25190840e-04, 3.13522585e-04, 4.89055878e-04, ...,\n",
       "         6.38576748e-05, 1.51719549e-04, 1.83751259e-03]],\n",
       "\n",
       "       [[1.07218155e-04, 8.44203605e-05, 7.20926851e-04, ...,\n",
       "         5.49410743e-06, 1.62863138e-03, 1.22820120e-03],\n",
       "        [2.41512520e-04, 6.90521047e-05, 3.95921874e-04, ...,\n",
       "         8.71878729e-05, 7.70738232e-04, 3.95830895e-04],\n",
       "        [1.30987071e-04, 1.00638084e-04, 6.47587702e-04, ...,\n",
       "         1.67643302e-04, 7.50067120e-04, 9.54770367e-04],\n",
       "        ...,\n",
       "        [2.15752935e-03, 2.93393987e-05, 1.42171339e-03, ...,\n",
       "         3.98446951e-04, 4.43629862e-04, 1.20063289e-03],\n",
       "        [5.37088723e-04, 1.82716249e-05, 5.35003725e-04, ...,\n",
       "         3.41485080e-04, 1.43029809e-03, 1.61124673e-03],\n",
       "        [1.41131968e-04, 1.49487814e-05, 1.38948765e-03, ...,\n",
       "         2.70983845e-04, 3.32316966e-04, 2.67567602e-03]]], dtype=float32)>"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model=ChatBotTrainer(encoder,decoder,name='chatbot_trainer')\n",
    "model.compile(\n",
    "    loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
    "    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),\n",
    "    weighted_metrics=['loss','accuracy']\n",
    ")\n",
    "model(_[:2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 5.7222 - accuracy: 0.0104     \n",
      "Epoch 1: val_loss improved from inf to 3.85731, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 38s 25s/step - loss: 5.6433 - accuracy: 0.0137 - val_loss: 3.8573 - val_accuracy: 0.1158\n",
      "Epoch 2/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 3.9921 - accuracy: 0.1386\n",
      "Epoch 2: val_loss improved from 3.85731 to 3.53857, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 28s 25s/step - loss: 3.9299 - accuracy: 0.1487 - val_loss: 3.5386 - val_accuracy: 0.2299\n",
      "Epoch 3/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 3.1027 - accuracy: 0.2351\n",
      "Epoch 3: val_loss improved from 3.53857 to 3.29378, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 29s 26s/step - loss: 2.9792 - accuracy: 0.2422 - val_loss: 3.2938 - val_accuracy: 0.2403\n",
      "Epoch 4/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 2.7268 - accuracy: 0.2907\n",
      "Epoch 4: val_loss improved from 3.29378 to 2.39274, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 29s 26s/step - loss: 2.7117 - accuracy: 0.2962 - val_loss: 2.3927 - val_accuracy: 0.3229\n",
      "Epoch 5/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 2.3293 - accuracy: 0.3674\n",
      "Epoch 5: val_loss improved from 2.39274 to 1.85684, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 32s 27s/step - loss: 2.2422 - accuracy: 0.3730 - val_loss: 1.8568 - val_accuracy: 0.4454\n",
      "Epoch 6/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 2.1871 - accuracy: 0.4019\n",
      "Epoch 6: val_loss did not improve from 1.85684\n",
      "2/2 [==============================] - 11s 6s/step - loss: 2.2527 - accuracy: 0.3958 - val_loss: 2.1417 - val_accuracy: 0.4470\n",
      "Epoch 7/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.9919 - accuracy: 0.4380\n",
      "Epoch 7: val_loss improved from 1.85684 to 1.82610, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 34s 28s/step - loss: 2.0411 - accuracy: 0.4317 - val_loss: 1.8261 - val_accuracy: 0.4255\n",
      "Epoch 8/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.7986 - accuracy: 0.4664\n",
      "Epoch 8: val_loss improved from 1.82610 to 1.74004, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 34s 28s/step - loss: 1.7946 - accuracy: 0.4662 - val_loss: 1.7400 - val_accuracy: 0.4583\n",
      "Epoch 9/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.6275 - accuracy: 0.5016\n",
      "Epoch 9: val_loss improved from 1.74004 to 1.57401, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 35s 29s/step - loss: 1.6067 - accuracy: 0.5020 - val_loss: 1.5740 - val_accuracy: 0.5442\n",
      "Epoch 10/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.5110 - accuracy: 0.5159\n",
      "Epoch 10: val_loss improved from 1.57401 to 1.24628, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 34s 28s/step - loss: 1.4511 - accuracy: 0.5174 - val_loss: 1.2463 - val_accuracy: 0.5142\n",
      "Epoch 11/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.4607 - accuracy: 0.5352\n",
      "Epoch 11: val_loss did not improve from 1.24628\n",
      "2/2 [==============================] - 12s 6s/step - loss: 1.4967 - accuracy: 0.5333 - val_loss: 1.3891 - val_accuracy: 0.5652\n",
      "Epoch 12/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.3088 - accuracy: 0.5736\n",
      "Epoch 12: val_loss did not improve from 1.24628\n",
      "2/2 [==============================] - 12s 6s/step - loss: 1.2618 - accuracy: 0.5828 - val_loss: 1.3992 - val_accuracy: 0.5584\n",
      "Epoch 13/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.2500 - accuracy: 0.5880\n",
      "Epoch 13: val_loss did not improve from 1.24628\n",
      "2/2 [==============================] - 13s 7s/step - loss: 1.2496 - accuracy: 0.5899 - val_loss: 1.3507 - val_accuracy: 0.5378\n",
      "Epoch 14/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.1871 - accuracy: 0.5993\n",
      "Epoch 14: val_loss did not improve from 1.24628\n",
      "2/2 [==============================] - 15s 8s/step - loss: 1.2048 - accuracy: 0.5965 - val_loss: 1.3987 - val_accuracy: 0.5551\n",
      "Epoch 15/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.0897 - accuracy: 0.6225\n",
      "Epoch 15: val_loss improved from 1.24628 to 1.15851, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 35s 30s/step - loss: 1.0756 - accuracy: 0.6297 - val_loss: 1.1585 - val_accuracy: 0.6408\n",
      "Epoch 16/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 1.0593 - accuracy: 0.6319\n",
      "Epoch 16: val_loss improved from 1.15851 to 1.01454, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 34s 28s/step - loss: 1.0750 - accuracy: 0.6351 - val_loss: 1.0145 - val_accuracy: 0.6758\n",
      "Epoch 17/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.9719 - accuracy: 0.6513\n",
      "Epoch 17: val_loss improved from 1.01454 to 0.90702, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 35s 30s/step - loss: 0.9586 - accuracy: 0.6505 - val_loss: 0.9070 - val_accuracy: 0.6733\n",
      "Epoch 18/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.9450 - accuracy: 0.6615\n",
      "Epoch 18: val_loss improved from 0.90702 to 0.62312, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 35s 28s/step - loss: 0.9508 - accuracy: 0.6629 - val_loss: 0.6231 - val_accuracy: 0.7152\n",
      "Epoch 19/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.9093 - accuracy: 0.6777\n",
      "Epoch 19: val_loss did not improve from 0.62312\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.9290 - accuracy: 0.6769 - val_loss: 1.2088 - val_accuracy: 0.6572\n",
      "Epoch 20/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.8513 - accuracy: 0.6866\n",
      "Epoch 20: val_loss did not improve from 0.62312\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.8435 - accuracy: 0.6914 - val_loss: 0.8376 - val_accuracy: 0.6947\n",
      "Epoch 21/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.7897 - accuracy: 0.7031\n",
      "Epoch 21: val_loss did not improve from 0.62312\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.7718 - accuracy: 0.7083 - val_loss: 0.8750 - val_accuracy: 0.6775\n",
      "Epoch 22/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.7584 - accuracy: 0.7111\n",
      "Epoch 22: val_loss did not improve from 0.62312\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.7354 - accuracy: 0.7147 - val_loss: 0.7470 - val_accuracy: 0.7328\n",
      "Epoch 23/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.7307 - accuracy: 0.7165\n",
      "Epoch 23: val_loss did not improve from 0.62312\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.7328 - accuracy: 0.7157 - val_loss: 0.7375 - val_accuracy: 0.6973\n",
      "Epoch 24/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.7268 - accuracy: 0.7244\n",
      "Epoch 24: val_loss did not improve from 0.62312\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.7487 - accuracy: 0.7225 - val_loss: 0.7659 - val_accuracy: 0.7182\n",
      "Epoch 25/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.6628 - accuracy: 0.7340\n",
      "Epoch 25: val_loss did not improve from 0.62312\n",
      "2/2 [==============================] - 14s 8s/step - loss: 0.6606 - accuracy: 0.7357 - val_loss: 0.8887 - val_accuracy: 0.6679\n",
      "Epoch 26/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.6573 - accuracy: 0.7398\n",
      "Epoch 26: val_loss did not improve from 0.62312\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.6542 - accuracy: 0.7421 - val_loss: 0.9965 - val_accuracy: 0.6748\n",
      "Epoch 27/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.6080 - accuracy: 0.7522\n",
      "Epoch 27: val_loss did not improve from 0.62312\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.5934 - accuracy: 0.7553 - val_loss: 0.6634 - val_accuracy: 0.7732\n",
      "Epoch 28/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.5961 - accuracy: 0.7609\n",
      "Epoch 28: val_loss improved from 0.62312 to 0.49036, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 38s 31s/step - loss: 0.5918 - accuracy: 0.7607 - val_loss: 0.4904 - val_accuracy: 0.7847\n",
      "Epoch 29/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.5648 - accuracy: 0.7697\n",
      "Epoch 29: val_loss did not improve from 0.49036\n",
      "2/2 [==============================] - 11s 5s/step - loss: 0.5566 - accuracy: 0.7726 - val_loss: 0.5248 - val_accuracy: 0.7660\n",
      "Epoch 30/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.5498 - accuracy: 0.7820\n",
      "Epoch 30: val_loss did not improve from 0.49036\n",
      "2/2 [==============================] - 17s 11s/step - loss: 0.5509 - accuracy: 0.7841 - val_loss: 0.5486 - val_accuracy: 0.7587\n",
      "Epoch 31/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.5357 - accuracy: 0.7815\n",
      "Epoch 31: val_loss did not improve from 0.49036\n",
      "2/2 [==============================] - 20s 11s/step - loss: 0.5385 - accuracy: 0.7774 - val_loss: 0.6815 - val_accuracy: 0.7599\n",
      "Epoch 32/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.5041 - accuracy: 0.7912 \n",
      "Epoch 32: val_loss did not improve from 0.49036\n",
      "2/2 [==============================] - 18s 7s/step - loss: 0.5114 - accuracy: 0.7904 - val_loss: 0.7003 - val_accuracy: 0.7400\n",
      "Epoch 33/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.5094 - accuracy: 0.7914\n",
      "Epoch 33: val_loss did not improve from 0.49036\n",
      "2/2 [==============================] - 11s 5s/step - loss: 0.5161 - accuracy: 0.7931 - val_loss: 0.7066 - val_accuracy: 0.7693\n",
      "Epoch 34/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.4587 - accuracy: 0.8054\n",
      "Epoch 34: val_loss improved from 0.49036 to 0.47291, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 33s 28s/step - loss: 0.4451 - accuracy: 0.8080 - val_loss: 0.4729 - val_accuracy: 0.8202\n",
      "Epoch 35/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.4678 - accuracy: 0.8053\n",
      "Epoch 35: val_loss improved from 0.47291 to 0.46739, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 30s 25s/step - loss: 0.4783 - accuracy: 0.8029 - val_loss: 0.4674 - val_accuracy: 0.8172\n",
      "Epoch 36/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.4448 - accuracy: 0.8140\n",
      "Epoch 36: val_loss did not improve from 0.46739\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.4543 - accuracy: 0.8111 - val_loss: 0.6211 - val_accuracy: 0.7593\n",
      "Epoch 37/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.4423 - accuracy: 0.8110\n",
      "Epoch 37: val_loss did not improve from 0.46739\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.4388 - accuracy: 0.8126 - val_loss: 0.8100 - val_accuracy: 0.7507\n",
      "Epoch 38/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.4093 - accuracy: 0.8259\n",
      "Epoch 38: val_loss did not improve from 0.46739\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.4059 - accuracy: 0.8263 - val_loss: 0.5436 - val_accuracy: 0.7903\n",
      "Epoch 39/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.4116 - accuracy: 0.8265\n",
      "Epoch 39: val_loss did not improve from 0.46739\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.4059 - accuracy: 0.8293 - val_loss: 0.5914 - val_accuracy: 0.7934\n",
      "Epoch 40/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3984 - accuracy: 0.8299\n",
      "Epoch 40: val_loss did not improve from 0.46739\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.3952 - accuracy: 0.8319 - val_loss: 0.5155 - val_accuracy: 0.7989\n",
      "Epoch 41/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3919 - accuracy: 0.8268\n",
      "Epoch 41: val_loss improved from 0.46739 to 0.33267, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 34s 30s/step - loss: 0.3950 - accuracy: 0.8261 - val_loss: 0.3327 - val_accuracy: 0.8250\n",
      "Epoch 42/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3710 - accuracy: 0.8351\n",
      "Epoch 42: val_loss did not improve from 0.33267\n",
      "2/2 [==============================] - 11s 6s/step - loss: 0.3713 - accuracy: 0.8347 - val_loss: 0.3899 - val_accuracy: 0.8420\n",
      "Epoch 43/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3569 - accuracy: 0.8464\n",
      "Epoch 43: val_loss did not improve from 0.33267\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.3542 - accuracy: 0.8460 - val_loss: 0.6216 - val_accuracy: 0.8310\n",
      "Epoch 44/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3442 - accuracy: 0.8498\n",
      "Epoch 44: val_loss did not improve from 0.33267\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.3352 - accuracy: 0.8515 - val_loss: 0.3940 - val_accuracy: 0.8406\n",
      "Epoch 45/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3449 - accuracy: 0.8503\n",
      "Epoch 45: val_loss did not improve from 0.33267\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.3518 - accuracy: 0.8490 - val_loss: 0.6558 - val_accuracy: 0.7959\n",
      "Epoch 46/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3365 - accuracy: 0.8528\n",
      "Epoch 46: val_loss did not improve from 0.33267\n",
      "2/2 [==============================] - 11s 5s/step - loss: 0.3347 - accuracy: 0.8548 - val_loss: 0.3817 - val_accuracy: 0.8319\n",
      "Epoch 47/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3254 - accuracy: 0.8585\n",
      "Epoch 47: val_loss did not improve from 0.33267\n",
      "2/2 [==============================] - 11s 6s/step - loss: 0.3323 - accuracy: 0.8587 - val_loss: 0.6199 - val_accuracy: 0.8211\n",
      "Epoch 48/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3040 - accuracy: 0.8649\n",
      "Epoch 48: val_loss did not improve from 0.33267\n",
      "2/2 [==============================] - 13s 6s/step - loss: 0.2956 - accuracy: 0.8691 - val_loss: 0.4242 - val_accuracy: 0.8260\n",
      "Epoch 49/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3126 - accuracy: 0.8660\n",
      "Epoch 49: val_loss did not improve from 0.33267\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.3153 - accuracy: 0.8659 - val_loss: 0.8541 - val_accuracy: 0.7926\n",
      "Epoch 50/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2974 - accuracy: 0.8739\n",
      "Epoch 50: val_loss did not improve from 0.33267\n",
      "2/2 [==============================] - 12s 7s/step - loss: 0.2994 - accuracy: 0.8737 - val_loss: 0.6176 - val_accuracy: 0.8262\n",
      "Epoch 51/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.3043 - accuracy: 0.8660\n",
      "Epoch 51: val_loss improved from 0.33267 to 0.25878, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 46s 40s/step - loss: 0.3100 - accuracy: 0.8656 - val_loss: 0.2588 - val_accuracy: 0.8830\n",
      "Epoch 52/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2830 - accuracy: 0.8757\n",
      "Epoch 52: val_loss did not improve from 0.25878\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.2839 - accuracy: 0.8746 - val_loss: 0.3063 - val_accuracy: 0.8618\n",
      "Epoch 53/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2851 - accuracy: 0.8689\n",
      "Epoch 53: val_loss did not improve from 0.25878\n",
      "2/2 [==============================] - 10s 5s/step - loss: 0.2894 - accuracy: 0.8677 - val_loss: 0.3811 - val_accuracy: 0.8524\n",
      "Epoch 54/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2671 - accuracy: 0.8807\n",
      "Epoch 54: val_loss did not improve from 0.25878\n",
      "2/2 [==============================] - 11s 5s/step - loss: 0.2592 - accuracy: 0.8815 - val_loss: 0.3261 - val_accuracy: 0.8814\n",
      "Epoch 55/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2688 - accuracy: 0.8806\n",
      "Epoch 55: val_loss did not improve from 0.25878\n",
      "2/2 [==============================] - 11s 6s/step - loss: 0.2647 - accuracy: 0.8792 - val_loss: 0.4454 - val_accuracy: 0.8674\n",
      "Epoch 56/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2587 - accuracy: 0.8845\n",
      "Epoch 56: val_loss improved from 0.25878 to 0.24759, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 42s 37s/step - loss: 0.2545 - accuracy: 0.8856 - val_loss: 0.2476 - val_accuracy: 0.8942\n",
      "Epoch 57/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2525 - accuracy: 0.8857\n",
      "Epoch 57: val_loss did not improve from 0.24759\n",
      "2/2 [==============================] - 12s 6s/step - loss: 0.2456 - accuracy: 0.8845 - val_loss: 0.2539 - val_accuracy: 0.9084\n",
      "Epoch 58/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2480 - accuracy: 0.8928\n",
      "Epoch 58: val_loss did not improve from 0.24759\n",
      "2/2 [==============================] - 13s 6s/step - loss: 0.2541 - accuracy: 0.8920 - val_loss: 0.2658 - val_accuracy: 0.8466\n",
      "Epoch 59/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2557 - accuracy: 0.8884\n",
      "Epoch 59: val_loss did not improve from 0.24759\n",
      "2/2 [==============================] - 12s 6s/step - loss: 0.2650 - accuracy: 0.8873 - val_loss: 0.2585 - val_accuracy: 0.8892\n",
      "Epoch 60/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2482 - accuracy: 0.8942\n",
      "Epoch 60: val_loss improved from 0.24759 to 0.23982, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 45s 40s/step - loss: 0.2508 - accuracy: 0.8939 - val_loss: 0.2398 - val_accuracy: 0.8939\n",
      "Epoch 61/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2278 - accuracy: 0.9004\n",
      "Epoch 61: val_loss did not improve from 0.23982\n",
      "2/2 [==============================] - 14s 8s/step - loss: 0.2297 - accuracy: 0.9006 - val_loss: 0.2467 - val_accuracy: 0.9058\n",
      "Epoch 62/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2276 - accuracy: 0.8976\n",
      "Epoch 62: val_loss improved from 0.23982 to 0.15475, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 45s 38s/step - loss: 0.2296 - accuracy: 0.8978 - val_loss: 0.1547 - val_accuracy: 0.9167\n",
      "Epoch 63/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2201 - accuracy: 0.8958\n",
      "Epoch 63: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.2203 - accuracy: 0.8947 - val_loss: 0.2987 - val_accuracy: 0.8846\n",
      "Epoch 64/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2138 - accuracy: 0.8993\n",
      "Epoch 64: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.2122 - accuracy: 0.8992 - val_loss: 0.4869 - val_accuracy: 0.8797\n",
      "Epoch 65/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2123 - accuracy: 0.9051\n",
      "Epoch 65: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.2113 - accuracy: 0.9070 - val_loss: 0.5685 - val_accuracy: 0.8453\n",
      "Epoch 66/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2183 - accuracy: 0.9029\n",
      "Epoch 66: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.2219 - accuracy: 0.9041 - val_loss: 0.4795 - val_accuracy: 0.8622\n",
      "Epoch 67/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2060 - accuracy: 0.9096\n",
      "Epoch 67: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.2052 - accuracy: 0.9102 - val_loss: 0.3051 - val_accuracy: 0.8906\n",
      "Epoch 68/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2089 - accuracy: 0.9052\n",
      "Epoch 68: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.2084 - accuracy: 0.9055 - val_loss: 0.2622 - val_accuracy: 0.9113\n",
      "Epoch 69/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.2023 - accuracy: 0.9057\n",
      "Epoch 69: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.2069 - accuracy: 0.9049 - val_loss: 0.3043 - val_accuracy: 0.8773\n",
      "Epoch 70/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1932 - accuracy: 0.9111\n",
      "Epoch 70: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1914 - accuracy: 0.9107 - val_loss: 0.5961 - val_accuracy: 0.8523\n",
      "Epoch 71/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1879 - accuracy: 0.9167\n",
      "Epoch 71: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1841 - accuracy: 0.9166 - val_loss: 0.3073 - val_accuracy: 0.8730\n",
      "Epoch 72/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1954 - accuracy: 0.9084\n",
      "Epoch 72: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.1945 - accuracy: 0.9078 - val_loss: 0.3331 - val_accuracy: 0.8963\n",
      "Epoch 73/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1840 - accuracy: 0.9173\n",
      "Epoch 73: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1874 - accuracy: 0.9170 - val_loss: 0.3596 - val_accuracy: 0.8852\n",
      "Epoch 74/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1925 - accuracy: 0.9162\n",
      "Epoch 74: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1951 - accuracy: 0.9165 - val_loss: 0.5873 - val_accuracy: 0.8752\n",
      "Epoch 75/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1793 - accuracy: 0.9180\n",
      "Epoch 75: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1726 - accuracy: 0.9191 - val_loss: 0.3152 - val_accuracy: 0.9151\n",
      "Epoch 76/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1878 - accuracy: 0.9161\n",
      "Epoch 76: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.1924 - accuracy: 0.9160 - val_loss: 0.1868 - val_accuracy: 0.9016\n",
      "Epoch 77/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1625 - accuracy: 0.9278\n",
      "Epoch 77: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1587 - accuracy: 0.9294 - val_loss: 0.6139 - val_accuracy: 0.8698\n",
      "Epoch 78/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1685 - accuracy: 0.9221\n",
      "Epoch 78: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.1704 - accuracy: 0.9229 - val_loss: 0.2777 - val_accuracy: 0.9137\n",
      "Epoch 79/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1713 - accuracy: 0.9231\n",
      "Epoch 79: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1702 - accuracy: 0.9232 - val_loss: 0.3409 - val_accuracy: 0.8949\n",
      "Epoch 80/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1635 - accuracy: 0.9286\n",
      "Epoch 80: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1621 - accuracy: 0.9301 - val_loss: 0.2284 - val_accuracy: 0.9139\n",
      "Epoch 81/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1628 - accuracy: 0.9262\n",
      "Epoch 81: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1637 - accuracy: 0.9262 - val_loss: 0.3800 - val_accuracy: 0.8871\n",
      "Epoch 82/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1564 - accuracy: 0.9324\n",
      "Epoch 82: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1540 - accuracy: 0.9334 - val_loss: 0.1915 - val_accuracy: 0.9336\n",
      "Epoch 83/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1513 - accuracy: 0.9296\n",
      "Epoch 83: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1517 - accuracy: 0.9304 - val_loss: 0.3214 - val_accuracy: 0.9070\n",
      "Epoch 84/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1530 - accuracy: 0.9293\n",
      "Epoch 84: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 13s 7s/step - loss: 0.1535 - accuracy: 0.9288 - val_loss: 0.1599 - val_accuracy: 0.9331\n",
      "Epoch 85/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1543 - accuracy: 0.9315\n",
      "Epoch 85: val_loss did not improve from 0.15475\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.1553 - accuracy: 0.9317 - val_loss: 0.2518 - val_accuracy: 0.9109\n",
      "Epoch 86/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1467 - accuracy: 0.9327\n",
      "Epoch 86: val_loss improved from 0.15475 to 0.13819, saving model to ckpt1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ckpt1\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2/2 [==============================] - 46s 40s/step - loss: 0.1418 - accuracy: 0.9359 - val_loss: 0.1382 - val_accuracy: 0.9510\n",
      "Epoch 87/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1474 - accuracy: 0.9342\n",
      "Epoch 87: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 15s 8s/step - loss: 0.1461 - accuracy: 0.9348 - val_loss: 0.2165 - val_accuracy: 0.9305\n",
      "Epoch 88/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1503 - accuracy: 0.9308\n",
      "Epoch 88: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.1477 - accuracy: 0.9321 - val_loss: 0.1822 - val_accuracy: 0.9190\n",
      "Epoch 89/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1403 - accuracy: 0.9385\n",
      "Epoch 89: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.1412 - accuracy: 0.9384 - val_loss: 0.4426 - val_accuracy: 0.9011\n",
      "Epoch 90/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1490 - accuracy: 0.9293\n",
      "Epoch 90: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.1498 - accuracy: 0.9283 - val_loss: 0.4667 - val_accuracy: 0.8838\n",
      "Epoch 91/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1355 - accuracy: 0.9373\n",
      "Epoch 91: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 15s 8s/step - loss: 0.1372 - accuracy: 0.9359 - val_loss: 0.1929 - val_accuracy: 0.9316\n",
      "Epoch 92/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1384 - accuracy: 0.9362\n",
      "Epoch 92: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 15s 8s/step - loss: 0.1405 - accuracy: 0.9354 - val_loss: 0.5629 - val_accuracy: 0.8943\n",
      "Epoch 93/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1336 - accuracy: 0.9418\n",
      "Epoch 93: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 15s 7s/step - loss: 0.1325 - accuracy: 0.9423 - val_loss: 0.5194 - val_accuracy: 0.8848\n",
      "Epoch 94/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1322 - accuracy: 0.9413\n",
      "Epoch 94: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.1277 - accuracy: 0.9429 - val_loss: 0.4517 - val_accuracy: 0.8997\n",
      "Epoch 95/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1335 - accuracy: 0.9405\n",
      "Epoch 95: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 15s 8s/step - loss: 0.1378 - accuracy: 0.9399 - val_loss: 0.2955 - val_accuracy: 0.8975\n",
      "Epoch 96/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1405 - accuracy: 0.9379\n",
      "Epoch 96: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 15s 8s/step - loss: 0.1440 - accuracy: 0.9349 - val_loss: 0.4010 - val_accuracy: 0.8941\n",
      "Epoch 97/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1142 - accuracy: 0.9514\n",
      "Epoch 97: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 15s 8s/step - loss: 0.1135 - accuracy: 0.9527 - val_loss: 0.5510 - val_accuracy: 0.8809\n",
      "Epoch 98/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1229 - accuracy: 0.9435\n",
      "Epoch 98: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 14s 7s/step - loss: 0.1218 - accuracy: 0.9442 - val_loss: 0.3778 - val_accuracy: 0.9104\n",
      "Epoch 99/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1265 - accuracy: 0.9414\n",
      "Epoch 99: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 16s 8s/step - loss: 0.1286 - accuracy: 0.9405 - val_loss: 0.5854 - val_accuracy: 0.8973\n",
      "Epoch 100/100\n",
      "2/2 [==============================] - ETA: 0s - loss: 0.1210 - accuracy: 0.9433\n",
      "Epoch 100: val_loss did not improve from 0.13819\n",
      "2/2 [==============================] - 15s 8s/step - loss: 0.1202 - accuracy: 0.9428 - val_loss: 0.1475 - val_accuracy: 0.9384\n"
     ]
    }
   ],
   "source": [
    "history=model.fit(\n",
    "    train_data,\n",
    "    epochs=100,\n",
    "    validation_data=val_data,\n",
    "    callbacks=[\n",
    "        tf.keras.callbacks.TensorBoard(log_dir='logs1'),\n",
    "        tf.keras.callbacks.ModelCheckpoint('ckpt1',verbose=1,save_best_only=True)\n",
    "    ]\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'plt' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32md:\\3 rd Year\\Y4S1\\Reserch\\Chatbot\\AllergyBot copy.ipynb Cell 24\u001b[0m in \u001b[0;36m1\n\u001b[1;32m----> <a href='vscode-notebook-cell:/d%3A/3%20rd%20Year/Y4S1/Reserch/Chatbot/AllergyBot%20copy.ipynb#X32sZmlsZQ%3D%3D?line=0'>1</a>\u001b[0m fig,ax\u001b[39m=\u001b[39mplt\u001b[39m.\u001b[39msubplots(nrows\u001b[39m=\u001b[39m\u001b[39m1\u001b[39m,ncols\u001b[39m=\u001b[39m\u001b[39m2\u001b[39m,figsize\u001b[39m=\u001b[39m(\u001b[39m20\u001b[39m,\u001b[39m5\u001b[39m))\n\u001b[0;32m      <a href='vscode-notebook-cell:/d%3A/3%20rd%20Year/Y4S1/Reserch/Chatbot/AllergyBot%20copy.ipynb#X32sZmlsZQ%3D%3D?line=1'>2</a>\u001b[0m ax[\u001b[39m0\u001b[39m]\u001b[39m.\u001b[39mplot(history\u001b[39m.\u001b[39mhistory[\u001b[39m'\u001b[39m\u001b[39mloss\u001b[39m\u001b[39m'\u001b[39m],label\u001b[39m=\u001b[39m\u001b[39m'\u001b[39m\u001b[39mloss\u001b[39m\u001b[39m'\u001b[39m,c\u001b[39m=\u001b[39m\u001b[39m'\u001b[39m\u001b[39mred\u001b[39m\u001b[39m'\u001b[39m)\n\u001b[0;32m      <a href='vscode-notebook-cell:/d%3A/3%20rd%20Year/Y4S1/Reserch/Chatbot/AllergyBot%20copy.ipynb#X32sZmlsZQ%3D%3D?line=2'>3</a>\u001b[0m ax[\u001b[39m0\u001b[39m]\u001b[39m.\u001b[39mplot(history\u001b[39m.\u001b[39mhistory[\u001b[39m'\u001b[39m\u001b[39mval_loss\u001b[39m\u001b[39m'\u001b[39m],label\u001b[39m=\u001b[39m\u001b[39m'\u001b[39m\u001b[39mval_loss\u001b[39m\u001b[39m'\u001b[39m,c \u001b[39m=\u001b[39m \u001b[39m'\u001b[39m\u001b[39mblue\u001b[39m\u001b[39m'\u001b[39m)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'plt' is not defined"
     ]
    }
   ],
   "source": [
    "fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,5))\n",
    "ax[0].plot(history.history['loss'],label='loss',c='red')\n",
    "ax[0].plot(history.history['val_loss'],label='val_loss',c = 'blue')\n",
    "ax[0].set_xlabel('Epochs')\n",
    "ax[1].set_xlabel('Epochs')\n",
    "ax[0].set_ylabel('Loss')\n",
    "ax[1].set_ylabel('Accuracy')\n",
    "ax[0].set_title('Loss Metrics')\n",
    "ax[1].set_title('Accuracy Metrics')\n",
    "ax[1].plot(history.history['accuracy'],label='accuracy')\n",
    "ax[1].plot(history.history['val_accuracy'],label='val_accuracy')\n",
    "ax[0].legend()\n",
    "ax[1].legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Found untraced functions such as _update_step_xla, lstm_cell_5_layer_call_fn, lstm_cell_5_layer_call_and_return_conditional_losses, lstm_cell_6_layer_call_fn, lstm_cell_6_layer_call_and_return_conditional_losses while saving (showing 5 of 5). These functions will not be directly callable after loading.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: models\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: models\\assets\n"
     ]
    }
   ],
   "source": [
    "model.load_weights('ckpt1')\n",
    "model.save('models',save_format='tf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Encoder layers:\n",
      "<keras.layers.core.embedding.Embedding object at 0x0000022C242AFA60>\n",
      "<keras.layers.normalization.layer_normalization.LayerNormalization object at 0x0000022C242AEB00>\n",
      "<keras.layers.rnn.lstm.LSTM object at 0x0000022C242AF280>\n",
      "---------------------\n",
      "Decoder layers: \n",
      "<keras.layers.core.embedding.Embedding object at 0x0000022C242AD9C0>\n",
      "<keras.layers.normalization.layer_normalization.LayerNormalization object at 0x0000022C242AEB90>\n",
      "<keras.layers.rnn.lstm.LSTM object at 0x0000022C242AE9B0>\n",
      "<keras.layers.normalization.layer_normalization.LayerNormalization object at 0x0000022C3EBA4790>\n",
      "<keras.layers.core.dense.Dense object at 0x0000022C3EBA77F0>\n",
      "---------------------\n"
     ]
    }
   ],
   "source": [
    "for idx,i in enumerate(model.layers):\n",
    "    print('Encoder layers:' if idx==0 else 'Decoder layers: ')\n",
    "    for j in i.layers:\n",
    "        print(j)\n",
    "    print('---------------------')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"chatbot_encoder\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " input_1 (InputLayer)        [(None, None)]            0         \n",
      "                                                                 \n",
      " encoder_embedding (Embeddin  (None, None, 256)        260096    \n",
      " g)                                                              \n",
      "                                                                 \n",
      " layer_normalization_7 (Laye  (None, None, 256)        512       \n",
      " rNormalization)                                                 \n",
      "                                                                 \n",
      " encoder_lstm (LSTM)         [(None, None, 256),       525312    \n",
      "                              (None, 256),                       \n",
      "                              (None, 256)]                       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 785,920\n",
      "Trainable params: 785,920\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "Model: \"chatbot_decoder\"\n",
      "__________________________________________________________________________________________________\n",
      " Layer (type)                   Output Shape         Param #     Connected to                     \n",
      "==================================================================================================\n",
      " input_4 (InputLayer)           [(None, None)]       0           []                               \n",
      "                                                                                                  \n",
      " decoder_embedding (Embedding)  (None, None, 256)    260096      ['input_4[0][0]']                \n",
      "                                                                                                  \n",
      " layer_normalization_7 (LayerNo  (None, None, 256)   512         ['decoder_embedding[0][0]']      \n",
      " rmalization)                                                                                     \n",
      "                                                                                                  \n",
      " input_2 (InputLayer)           [(None, 256)]        0           []                               \n",
      "                                                                                                  \n",
      " input_3 (InputLayer)           [(None, 256)]        0           []                               \n",
      "                                                                                                  \n",
      " decoder_lstm (LSTM)            [(None, None, 256),  525312      ['layer_normalization_7[1][0]',  \n",
      "                                 (None, 256),                     'input_2[0][0]',                \n",
      "                                 (None, 256)]                     'input_3[0][0]']                \n",
      "                                                                                                  \n",
      " decoder_dense (Dense)          (None, None, 1016)   261112      ['decoder_lstm[0][0]']           \n",
      "                                                                                                  \n",
      "==================================================================================================\n",
      "Total params: 1,047,032\n",
      "Trainable params: 1,047,032\n",
      "Non-trainable params: 0\n",
      "__________________________________________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "class ChatBot(tf.keras.models.Model):\n",
    "    def __init__(self,base_encoder,base_decoder,*args,**kwargs):\n",
    "        super().__init__(*args,**kwargs)\n",
    "        self.encoder,self.decoder=self.build_inference_model(base_encoder,base_decoder)\n",
    "\n",
    "    def build_inference_model(self,base_encoder,base_decoder):\n",
    "        encoder_inputs=tf.keras.Input(shape=(None,))\n",
    "        x=base_encoder.layers[0](encoder_inputs)\n",
    "        x=base_encoder.layers[1](x)\n",
    "        x,encoder_state_h,encoder_state_c=base_encoder.layers[2](x)\n",
    "        encoder=tf.keras.models.Model(inputs=encoder_inputs,outputs=[encoder_state_h,encoder_state_c],name='chatbot_encoder')\n",
    "\n",
    "        decoder_input_state_h=tf.keras.Input(shape=(lstm_cells,))\n",
    "        decoder_input_state_c=tf.keras.Input(shape=(lstm_cells,))\n",
    "        decoder_inputs=tf.keras.Input(shape=(None,))\n",
    "        x=base_decoder.layers[0](decoder_inputs)\n",
    "        x=base_encoder.layers[1](x)\n",
    "        x,decoder_state_h,decoder_state_c=base_decoder.layers[2](x,initial_state=[decoder_input_state_h,decoder_input_state_c])\n",
    "        decoder_outputs=base_decoder.layers[-1](x)\n",
    "        decoder=tf.keras.models.Model(\n",
    "            inputs=[decoder_inputs,[decoder_input_state_h,decoder_input_state_c]],\n",
    "            outputs=[decoder_outputs,[decoder_state_h,decoder_state_c]],name='chatbot_decoder'\n",
    "        )\n",
    "        return encoder,decoder\n",
    "\n",
    "    def summary(self):\n",
    "        self.encoder.summary()\n",
    "        self.decoder.summary()\n",
    "\n",
    "    def softmax(self,z):\n",
    "        return np.exp(z)/sum(np.exp(z))\n",
    "\n",
    "    def sample(self,conditional_probability,temperature=0.5):\n",
    "        conditional_probability = np.asarray(conditional_probability).astype(\"float64\")\n",
    "        conditional_probability = np.log(conditional_probability) / temperature\n",
    "        reweighted_conditional_probability = self.softmax(conditional_probability)\n",
    "        probas = np.random.multinomial(1, reweighted_conditional_probability, 1)\n",
    "        return np.argmax(probas)\n",
    "\n",
    "    def preprocess(self,text):\n",
    "        text=clean_text(text)\n",
    "        seq=np.zeros((1,max_sequence_length),dtype=np.int32)\n",
    "        for i,word in enumerate(text.split()):\n",
    "            seq[:,i]=sequences2ids(word).numpy()[0]\n",
    "        return seq\n",
    "    \n",
    "    def postprocess(self,text):\n",
    "        text=re.sub(' - ','-',text.lower())\n",
    "        text=re.sub(' [.] ','. ',text)\n",
    "        text=re.sub(' [1] ','1',text)\n",
    "        text=re.sub(' [2] ','2',text)\n",
    "        text=re.sub(' [3] ','3',text)\n",
    "        text=re.sub(' [4] ','4',text)\n",
    "        text=re.sub(' [5] ','5',text)\n",
    "        text=re.sub(' [6] ','6',text)\n",
    "        text=re.sub(' [7] ','7',text)\n",
    "        text=re.sub(' [8] ','8',text)\n",
    "        text=re.sub(' [9] ','9',text)\n",
    "        text=re.sub(' [0] ','0',text)\n",
    "        text=re.sub(' [,] ',', ',text)\n",
    "        text=re.sub(' [?] ','? ',text)\n",
    "        text=re.sub(' [!] ','! ',text)\n",
    "        text=re.sub(' [$] ','$ ',text)\n",
    "        text=re.sub(' [&] ','& ',text)\n",
    "        text=re.sub(' [/] ','/ ',text)\n",
    "        text=re.sub(' [:] ',': ',text)\n",
    "        text=re.sub(' [;] ','; ',text)\n",
    "        text=re.sub(' [*] ','* ',text)\n",
    "        text=re.sub(' [\\'] ','\\'',text)\n",
    "        text=re.sub(' [\\\"] ','\\\"',text)\n",
    "        return text\n",
    "\n",
    "    def call(self,text,config=None):\n",
    "        input_seq=self.preprocess(text)\n",
    "        states=self.encoder(input_seq,training=False)\n",
    "        target_seq=np.zeros((1,1))\n",
    "        target_seq[:,:]=sequences2ids(['<start>']).numpy()[0][0]\n",
    "        stop_condition=False\n",
    "        decoded=[]\n",
    "        while not stop_condition:\n",
    "            decoder_outputs,new_states=self.decoder([target_seq,states],training=False)\n",
    "#             index=tf.argmax(decoder_outputs[:,-1,:],axis=-1).numpy().item()\n",
    "            index=self.sample(decoder_outputs[0,0,:]).item()\n",
    "            word=ids2sequences([index])\n",
    "            if word=='<end> ' or len(decoded)>=max_sequence_length:\n",
    "                stop_condition=True\n",
    "            else:\n",
    "                decoded.append(index)\n",
    "                target_seq=np.zeros((1,1))\n",
    "                target_seq[:,:]=index\n",
    "                states=new_states\n",
    "        return self.postprocess(ids2sequences(decoded))\n",
    "\n",
    "chatbot=ChatBot(model.encoder,model.decoder,name='chatbot')\n",
    "chatbot.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pydot in c:\\users\\nuwanga wijamuni\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (1.4.2)Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: You are using pip version 22.0.4; however, version 23.1.2 is available.\n",
      "You should consider upgrading via the 'c:\\Users\\Nuwanga Wijamuni\\AppData\\Local\\Programs\\Python\\Python310\\python.exe -m pip install --upgrade pip' command.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Requirement already satisfied: pyparsing>=2.1.4 in c:\\users\\nuwanga wijamuni\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from pydot) (3.0.9)\n"
     ]
    }
   ],
   "source": [
    "pip install pydot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model to work.\n"
     ]
    }
   ],
   "source": [
    "tf.keras.utils.plot_model(chatbot.encoder,to_file='encoder.png',show_shapes=True,show_layer_activations=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model to work.\n"
     ]
    }
   ],
   "source": [
    "tf.keras.utils.plot_model(chatbot.decoder,to_file='decoder.png',show_shapes=True,show_layer_activations=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_conversation(texts):\n",
    "    for text in texts:\n",
    "        print(f'Patient: {text}')\n",
    "        print(f'DocBot: {chatbot(text)}')\n",
    "        print('========================')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Patient: Hellow Doctor\n",
      "DocBot: how can i help you today? \n",
      "========================\n",
      "Patient: I haven't noticed any significant swelling\n",
      "DocBot: i appreciate it. i'll start by examining your neck for any signs of redness, hives, or swelling that might indicate an allergic reaction. \n",
      "========================\n",
      "Patient:  I'm just a bit concerned about the severity of the reaction\n",
      "DocBot: it's important to take these symptoms seriously, as course, or even cross contamination with other allergens could be responsible for your symptoms. anaphylaxis can occur even with small amounts of allergens. it's \n",
      "========================\n"
     ]
    }
   ],
   "source": [
    "print_conversation([\n",
    "    'Hellow Doctor',\n",
    "    '''I haven't noticed any significant swelling''',\n",
    "    ''' I'm just a bit concerned about the severity of the reaction'''\n",
    "\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'tf' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32md:\\3 rd Year\\Y4S1\\Reserch\\Chatbot\\AllergyBot copy.ipynb Cell 33\u001b[0m in \u001b[0;36m1\n\u001b[1;32m----> <a href='vscode-notebook-cell:/d%3A/3%20rd%20Year/Y4S1/Reserch/Chatbot/AllergyBot%20copy.ipynb#X44sZmlsZQ%3D%3D?line=0'>1</a>\u001b[0m \u001b[39mclass\u001b[39;00m \u001b[39mRealTimeChatBot\u001b[39;00m(tf\u001b[39m.\u001b[39mkeras\u001b[39m.\u001b[39mmodels\u001b[39m.\u001b[39mModel):\n\u001b[0;32m      <a href='vscode-notebook-cell:/d%3A/3%20rd%20Year/Y4S1/Reserch/Chatbot/AllergyBot%20copy.ipynb#X44sZmlsZQ%3D%3D?line=1'>2</a>\u001b[0m     \u001b[39mdef\u001b[39;00m \u001b[39m__init__\u001b[39m(\u001b[39mself\u001b[39m,base_encoder,base_decoder,\u001b[39m*\u001b[39margs,\u001b[39m*\u001b[39m\u001b[39m*\u001b[39mkwargs):\n\u001b[0;32m      <a href='vscode-notebook-cell:/d%3A/3%20rd%20Year/Y4S1/Reserch/Chatbot/AllergyBot%20copy.ipynb#X44sZmlsZQ%3D%3D?line=2'>3</a>\u001b[0m         \u001b[39msuper\u001b[39m()\u001b[39m.\u001b[39m\u001b[39m__init__\u001b[39m(\u001b[39m*\u001b[39margs,\u001b[39m*\u001b[39m\u001b[39m*\u001b[39mkwargs)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'tf' is not defined"
     ]
    }
   ],
   "source": [
    "class RealTimeChatBot(tf.keras.models.Model):\n",
    "    def __init__(self,base_encoder,base_decoder,*args,**kwargs):\n",
    "        super().__init__(*args,**kwargs)\n",
    "        self.encoder,self.decoder=self.build_inference_model(base_encoder,base_decoder)\n",
    "\n",
    "    def build_inference_model(self,base_encoder,base_decoder):\n",
    "        encoder_inputs=tf.keras.Input(shape=(None,))\n",
    "        x=base_encoder.layers[0](encoder_inputs)\n",
    "        x=base_encoder.layers[1](x)\n",
    "        x,encoder_state_h,encoder_state_c=base_encoder.layers[2](x)\n",
    "        encoder=tf.keras.models.Model(inputs=encoder_inputs,outputs=[encoder_state_h,encoder_state_c],name='chatbot_encoder')\n",
    "\n",
    "        decoder_input_state_h=tf.keras.Input(shape=(lstm_cells,))\n",
    "        decoder_input_state_c=tf.keras.Input(shape=(lstm_cells,))\n",
    "        decoder_inputs=tf.keras.Input(shape=(None,))\n",
    "        x=base_decoder.layers[0](decoder_inputs)\n",
    "        x=base_encoder.layers[1](x)\n",
    "        x,decoder_state_h,decoder_state_c=base_decoder.layers[2](x,initial_state=[decoder_input_state_h,decoder_input_state_c])\n",
    "        decoder_outputs=base_decoder.layers[-1](x)\n",
    "        decoder=tf.keras.models.Model(\n",
    "            inputs=[decoder_inputs,[decoder_input_state_h,decoder_input_state_c]],\n",
    "            outputs=[decoder_outputs,[decoder_state_h,decoder_state_c]],name='chatbot_decoder'\n",
    "        )\n",
    "        return encoder,decoder\n",
    "\n",
    "    def summary(self):\n",
    "        self.encoder.summary()\n",
    "        self.decoder.summary()\n",
    "\n",
    "    def softmax(self,z):\n",
    "        return np.exp(z)/sum(np.exp(z))\n",
    "\n",
    "    def sample(self,conditional_probability,temperature=0.5):\n",
    "        conditional_probability = np.asarray(conditional_probability).astype(\"float64\")\n",
    "        conditional_probability = np.log(conditional_probability) / temperature\n",
    "        reweighted_conditional_probability = self.softmax(conditional_probability)\n",
    "        probas = np.random.multinomial(1, reweighted_conditional_probability, 1)\n",
    "        return np.argmax(probas)\n",
    "\n",
    "    def preprocess(self,text):\n",
    "        text=clean_text(text)\n",
    "        seq=np.zeros((1,max_sequence_length),dtype=np.int32)\n",
    "        for i,word in enumerate(text.split()):\n",
    "            seq[:,i]=sequences2ids(word).numpy()[0]\n",
    "        return seq\n",
    "    \n",
    "    def postprocess(self,text):\n",
    "        text=re.sub(' - ','-',text.lower())\n",
    "        text=re.sub(' [.] ','. ',text)\n",
    "        text=re.sub(' [1] ','1',text)\n",
    "        text=re.sub(' [2] ','2',text)\n",
    "        text=re.sub(' [3] ','3',text)\n",
    "        text=re.sub(' [4] ','4',text)\n",
    "        text=re.sub(' [5] ','5',text)\n",
    "        text=re.sub(' [6] ','6',text)\n",
    "        text=re.sub(' [7] ','7',text)\n",
    "        text=re.sub(' [8] ','8',text)\n",
    "        text=re.sub(' [9] ','9',text)\n",
    "        text=re.sub(' [0] ','0',text)\n",
    "        text=re.sub(' [,] ',', ',text)\n",
    "        text=re.sub(' [?] ','? ',text)\n",
    "        text=re.sub(' [!] ','! ',text)\n",
    "        text=re.sub(' [$] ','$ ',text)\n",
    "        text=re.sub(' [&] ','& ',text)\n",
    "        text=re.sub(' [/] ','/ ',text)\n",
    "        text=re.sub(' [:] ',': ',text)\n",
    "        text=re.sub(' [;] ','; ',text)\n",
    "        text=re.sub(' [*] ','* ',text)\n",
    "        text=re.sub(' [\\'] ','\\'',text)\n",
    "        text=re.sub(' [\\\"] ','\\\"',text)\n",
    "        return text\n",
    "\n",
    "    def call(self,text,config=None):\n",
    "        input_seq=self.preprocess(text)\n",
    "        states=self.encoder(input_seq,training=False)\n",
    "        target_seq=np.zeros((1,1))\n",
    "        target_seq[:,:]=sequences2ids(['<start>']).numpy()[0][0]\n",
    "        stop_condition=False\n",
    "        decoded=[]\n",
    "        while not stop_condition:\n",
    "            decoder_outputs,new_states=self.decoder([target_seq,states],training=False)\n",
    "#             index=tf.argmax(decoder_outputs[:,-1,:],axis=-1).numpy().item()\n",
    "            index=self.sample(decoder_outputs[0,0,:]).item()\n",
    "            word=ids2sequences([index])\n",
    "            if word=='<end> ' or len(decoded)>=max_sequence_length:\n",
    "                stop_condition=True\n",
    "            else:\n",
    "                decoded.append(index)\n",
    "                target_seq=np.zeros((1,1))\n",
    "                target_seq[:,:]=index\n",
    "                states=new_states\n",
    "        return self.postprocess(ids2sequences(decoded))\n",
    "\n",
    "    def chat(self):\n",
    "        print(\"Bot: Hello! How can I assist you today?\")\n",
    "        while True:\n",
    "            user_input = input('User: ')\n",
    "            if user_input.lower() in ['quit', 'exit', 'bye']:\n",
    "                print('Bot: Goodbye! Take care.')\n",
    "                break\n",
    "            response = self.call(user_input)\n",
    "            print(f'Bot: {response}')\n",
    "\n",
    "# Initialize and summarize chatbot\n",
    "chatbot=RealTimeChatBot(model.encoder,model.decoder,name='chatbot')\n",
    "chatbot.summary()\n",
    "\n",
    "# Start a real-time chat with the chatbot\n",
    "chatbot.chat()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
