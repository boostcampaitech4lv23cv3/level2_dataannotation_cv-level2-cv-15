# πνλ‘μ νΈ κ°μ

<img src="./image/title.png" />

OCR (Optimal Character Recognition) κΈ°μ μ μ¬λμ΄ μ§μ  μ°κ±°λ μ΄λ―Έμ§ μμ μλ λ¬Έμλ₯Ό μ»μ λ€μ μ΄λ₯Ό μ»΄ν¨ν°κ° μΈμν  μ μλλ‘ νλ κΈ°μ λ‘, μ»΄ν¨ν° λΉμ  λΆμΌμμ νμ¬ λλ¦¬ μ°μ΄λ λνμ μΈ κΈ°μ  μ€ νλμλλ€. OCR taskλ κΈμ κ²μΆ (text detection), κΈμ μΈμ (text recognition), μ λ ¬κΈ° (Serializer) λ±μ λͺ¨λλ‘ μ΄λ£¨μ΄μ Έ μμ΅λλ€. λ³Έ λνμμλ μ μ½λ νκ²½μμ 'κΈμ κ²μΆ' task λ§μ ν΄κ²°νκ² λ©λλ€.

<p align="center"><img src="./image/OCR.png" alt="trash" width="40%" height="40%" /></p>



- λνμ λͺ©ν : μμ§μ λ°μ΄ν°λ₯Ό μ£Όμ΄μ§ λͺ¨λΈμ μ κ³΅νμ¬ λμ μ νλλ₯Ό λ³΄μ΄λλ‘ νμ΅μν€λ κ²

- μ μ½μ¬ν­

  1. λͺ¨λΈ - Text Detector λ€νΈμν¬ μ€ νλμΈ EAST
  2. Loss, νκ°μ§ν

- μμ ν  μ μλ κ²

  1. Data - λ€μν λ°μ΄ν°λ₯Ό κ°μ Έμμ λͺ¨λΈμ νμ΅μν¨λ€.
  2. Dataset μ½λ - λ°μ΄ν°μ λ€μμ±μ μν΄ agumentation μ μ© κ°λ₯νλ€.
  3. Train μ½λ - validation μ½λ μΆκ°, νμ΄νΌνλΌλ―Έν° νλ κ°λ₯
  4. Inference μ½λ - TTAμ μ©κ°λ₯

- νκ° λ°μ΄ν°

  <p align="center"><img src="./image/public.png" alt="trash" width="80%" height="80%" /></p>



### νκ°μ§ν

**DetEval**

μ΄λ―Έμ§ λ λ²¨μμ μ λ΅ λ°μ€κ° μ¬λ¬κ° μ‘΄μ¬νκ³ , μμΈ‘ν λ°μ€κ° μ¬λ¬κ°κ° μμ κ²½μ°, λ°μ€λΌλ¦¬μ λ€μ€ λ§€μΉ­μ νμ©νμ¬ μ μλ₯Ό μ£Όλ νκ°λ°©λ² μ€ νλ

1. **λͺ¨λ  μ λ΅/μμΈ‘λ°μ€λ€μ λν΄μ Area Recall, Area Precisionμ λ―Έλ¦¬ κ³μ°**

   <p align="center"><img src="./image/eval.png" alt="trash" width="80%" height="80%" /></p>





# β νλ‘μ νΈ μν μ μ°¨ λ° λ°©λ²

λν κ·μΉμ λ°λΌμ Model, loss, detectλ±μ μμ ν  μ μκΈ° λλ¬Έμ μ±λ₯μ ν₯μ μν¬ μ μλ λ°©λ²μΌλ‘ Data μΆκ°μ augmentation, Hyperparameterλ₯Ό μ‘°μ νλ λ°©λ²μΌλ‘ μ§ννμλ€.

### Dataset

- ICDAR17 Korean
- ICDAR17 English & Korean
- ICDAR19 English & Korean
- AIHub
  - νκ΅­μ΄ κΈμμ²΄ μ΄λ―Έμ§(Text in the Wild)
    - νμ§ν(500μ₯)
    - κ°ν(500μ₯)
  - μΌμΈ μ€μ  μ΄¬μ νκΈ μ΄λ―Έμ§
    - μ±νμ§(500μ₯)

### Experiment

- batch_size

  - 12
  - 16
  - 32

- Scheduler

  - StepLR
  - MultiStepLR
  - CosineAnnealingLR
  - CosineAnnealingWarmRestart
  - ReduceLROnPlateau

- Epoch

  - 150
  - 200
  - 300

- Optimizer

  - Adam

- Augmentation

  ColorJitter, ISONoise, RandomGamma, HueSaturationValue, ChannelShuffle, CLAHE, RandomBrightnessContrast, Emboss, Sharpen, Equalize





# π νλ‘μ νΈ κ²°κ³Ό

- loss

  <p align="center"><img src="./image/loss.png" alt="trash" width="60%" height="60%" /></p>

- score

  <p align="center"><img src="./image/public.png" alt="trash" width="100%" height="100%" /></p>

  



# π¨βπ¨βπ§βπ¦ νμ μκ°


|         [ν©μμ](https://github.com/soonyoung-hwang)         |            [μμμ€](https://github.com/won-joon)             |              [μ΄νμ ](https://github.com/SS-hj)              |             [κΉλν](https://github.com/DHKim95)             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![SY_image](https://avatars.githubusercontent.com/u/78343941?v=4) | ![WJ_image](https://avatars.githubusercontent.com/u/59519591?v=4) | ![HJ_image](https://avatars.githubusercontent.com/u/54202082?v=4) | ![DH_image](https://avatars.githubusercontent.com/u/68861542?v=4) |

