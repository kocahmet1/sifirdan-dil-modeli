# Sıfırdan Dil Modeli

Türkçe bir GPT'yi sıfırdan, satır satır yazdığımız YouTube serisinin defterleri. Her bölümde kodu birlikte yazıyor, her satırın ne
yaptığını konuşuyor ve gerçek bir Türkçe metin üzerinde çalıştırıyoruz.

**Kurulum yok, ücret yok: yalnızca bir tarayıcı.** Aşağıdaki "Colab'da aç" linkine tıkla; defter Google Colab'da açılır. İlk hücreler
gerekenleri bu depodan kendisi indirir. Hücreleri sırayla çalıştırmak için `Shift + Enter`.

## Bölümler

| # | Bölüm | Defter |
| --- | --- | --- |
| E01 | Metinden sayılara: ilk tokenizer'ımız | [Colab'da aç](https://colab.research.google.com/github/kocahmet1/sifirdan-dil-modeli/blob/main/defter/E01_metinden_sayilara.ipynb) |
| E02 | BPE: kelimeyi parçalara bölmek | [Colab'da aç](https://colab.research.google.com/github/kocahmet1/sifirdan-dil-modeli/blob/main/defter/E02_bpe.ipynb) |
| E03 | Kayan pencere: girdi ve hedef | [Colab'da aç](https://colab.research.google.com/github/kocahmet1/sifirdan-dil-modeli/blob/main/defter/E03_kayan_pencere.ipynb) |
| E04 | Gömme: sayılardan vektörlere | [Colab'da aç](https://colab.research.google.com/github/kocahmet1/sifirdan-dil-modeli/blob/main/defter/E04_gomme.ipynb) |
| E05 | Dikkat: kelimeler birbirine bakıyor | [Colab'da aç](https://colab.research.google.com/github/kocahmet1/sifirdan-dil-modeli/blob/main/defter/E05_dikkat.ipynb) |
| E06 | Dikkat II: sorgu, anahtar, değer | [Colab'da aç](https://colab.research.google.com/github/kocahmet1/sifirdan-dil-modeli/blob/main/defter/E06_sorgu_anahtar_deger.ipynb) |
| E07 | Dikkat III: maske ve çok başlı dikkat | [Colab'da aç](https://colab.research.google.com/github/kocahmet1/sifirdan-dil-modeli/blob/main/defter/E07_cok_basli_dikkat.ipynb) |
| E08 | GPT'nin parçaları | [Colab'da aç](https://colab.research.google.com/github/kocahmet1/sifirdan-dil-modeli/blob/main/defter/E08_gpt_parcalari.ipynb) |
| E09 | GPT'yi birleştirmek: ilk metin | [Colab'da aç](https://colab.research.google.com/github/kocahmet1/sifirdan-dil-modeli/blob/main/defter/E09_gpt.ipynb) |

## Depoda ne var

| Dosya | Ne |
| --- | --- |
| `defter/` | Bölümlerin defterleri (çıktısız; videoda yazdığımız kodun aynısı) |
| `onceki_bolumler.py` | Önceki bölümlerde yazdığımız kod tek dosyada: tokenizer, veri yükleyici, dikkat katmanları, GPT'nin parçaları. 3. bölümden itibaren her defter bunu indirip içinden alır |
| `veri/kasagi.txt` | Ömer Seyfettin, "Kaşağı": serinin ilk bölümlerinde okuduğumuz öykü (6.305 karakter) |
| `veri/cirak_tokenizer.json` | Çırak'ın tokenizer'ı: bayt düzeyli BPE, 4.096 parça |
| `veri/cirak_gomme.pt` | Çırak'ın gömme tablosu (4.096 × 256) |
| `veri/cirak_bas.pt` | Çırak'ın 3. bloğundaki 6. dikkat başı |
| `veri/cirak_blok2.pt` | Çırak'ın 2. bloğunun dikkat ağırlıkları |
| `veri/cirak_gpt.pt` | Çırak'ın bütün ağırlıkları, kitaptaki `GPTModel`'in katman adlarıyla |

**Çırak** serinin kendi küçük Türkçe modeli: Türkçe Vikipedi'de eğitildi. 2. bölümden itibaren defterler, yazdığımız parçaları onun
eğitilmiş tokenizer'ı ve ağırlıklarıyla da deniyor; 9. bölümde bütün model birleşip ilk Türkçe cümlelerini yazıyor.

## Kaynak ve lisans

Defterlerdeki ve `onceki_bolumler.py`'deki kod, Sebastian Raschka'nın *Build a Large Language Model (From Scratch)* kitabının
kodunun ([rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch), Apache License 2.0) Türkçe metin için
uyarlamasıdır. Başlıca değişiklikler: Türkçe metinler ve açıklamalar, GPT-2'nin tokenizer'ı yerine serinin kendi bayt düzeyli BPE'si
(Çırak'ın tokenizer'ı) ve kitaptaki GPT-2 boyutu yerine Çırak'ın boyunda bir model. Bu depodaki kod da Apache License 2.0 ile
paylaşılıyor (`LICENSE`).

- `veri/kasagi.txt`: Ömer Seyfettin (1884–1920), "Kaşağı"; kamu malı.
- `veri/cirak_*`: Türkçe Vikipedi metinleriyle (CC BY-SA 4.0, Vikipedi katkıcıları) eğitilmiş Çırak modelinden.
