## Animal-classification-with-gradio

Bu proje, bir yapay zeka modelinin baştan sona tamamını içerir. Proje, veri ön işleme, özellik çıkarımı, veri kümesi kümeleme, veri ayırma, model eğitimi ve web arayüzü ile modeli sunma adımlarını kapsamaktadır.

# Hakkında
Bu proje, bir yapay zeka modelinin veri ön işleme aşamasından, modelin eğitilmesine ve sonrasında web arayüzü ile kullanıcıya sunulmasına kadar tüm adımları içermektedir. Proje, mobilenetv3_large_100 tabanlı bir model kullanarak çeşitli hayvan türlerini sınıflandırmak için tasarlanmıştır.

# Özellikler
-Resimlerin ön işlenmesi ve ilgisiz kısımlarının kesilmesi
-Resimlerden vektörlerin çıkarılması ve embeddinglerin oluşturulması
-PCA ve KMeans kullanarak veri kümesinin kümeleme analizi
-Veri kümesinin eğitim ve test olarak ayrılması
-Mobilenetv3_large_100 modeli ile eğitilmesi
-Gradio kullanarak web arayüzü ile modelin sunumu

## Kurulum
----------------------

# Gereksinimler
----------------------

Projeyi yerel makinenize kurmadan önce aşağıdaki araçların kurulu olduğundan emin olun:

-Python 3.8+

-Pytorch (mümkünse cuda ile)

-Torchvision

-timm

tqdm
scipy
sklearn
opencv
PIL
matplotlib
gradio
pandas
numpy
img2vec_pytorch

## Adımlar
----------------------

# Repoyu klonlayın

    git clone https://github.com/baloglu321/Animal-classification-with-gradio.git


# Proje dizinine geçin
    
    cd Animal-classification-with-gradio

## Kullanım
----------------------

Proje, aşağıdaki adımları takip ederek kullanılabilir:

1. Veri Ön İşleme
Veri kümesini alıp ilgisiz kısımları kesin:

        python preprocess.py

2. Özellik Çıkarımı
Resimlerin embeddinglerini çıkarın ve kaydedin:

        python get_embedings.py

3. Veri Kümesi Kümeleme
Embeddingleri okuyun ve PCA ve KMeans algoritmalarını kullanarak veri setini kümeleyin:

        python clustring.py

4. Veri Ayırma
Veri setini %80-20 oranında eğitim ve test olarak ayırın:

        python data_splitter.py

5. Model Eğitimi
Modeli eğitin:

        python train.py

6. Web Arayüzü ile Model Sunumu
Modeli web arayüzü ile kullanıma sunun:

        python gradio_infer.py


## Train için dosya yapısı
----------------------

├───model_dataset
│   ├───test
│   │   ├───antelope
│   │   ├───badger
│   │   ├───bat
│   │   ├───bear
.   .   .
.   .   .
.   .   .
│   └───train
│       ├───antelope
│       ├───badger
│       ├───bat
│       ├───bear
.    .  .

## Notlar
----------------------

Training için yukardaki dosya yapısı ile verilerin verilmesi ve Train.py dosyasının çalıştırılması yeterlidir. 
Fakat clustring vb. algoritmaların çaçlıştılması için sırsıyla:
1-Raw dataset oluştulmalı:
-Bulunulan dizinde "raw_images" isminde bir dizin oluşturularak aşağıdaki görülen şekildeki dosya yapısıyla önişlenecekd data verilmeli:
├───raw_images
│   ├───antelope
│   ├───badger
│   ├───bat
│   ├───bear

-Sonrasında "preprocess.py dosyası çalıştırılır. Bu çalıştırıldığında "processed_images" isimli bir dizin oluşur ve kesilen ve filtrelen görüntüler buraya kaydedilir. Modele oluşturulacak girdiye göre maximum ve minimum görüntü boyutlarını kod içerisinden belirleyebilrisiniz. Varsayılan: max:1024 min:224
-Resimler içerisinde ilgisiz dataların belirlenmesi için vektörler üzerinden kmeans ile clustring işlemi uygulamak için öncelikle "get_embeddings.py" çalıştırılır. Bu algoritma "embeddings" isimli klasörün içerisine resimlerin vektörlerini csv formatında kaydeder. 
-"clustring.py" dosyası çalıştırılarak kaydedilen vektörler okunur ve pca ve kmeans algoritmaları ile veri setinde ilgili görülen datalar "clusters" isimli dizine her class için ayrılarak kaydedilir. Bu clusterlardan ilgisiz görülenler manuel olarak silinmelidir. Birden fazla cluster bir araya getirilirse tekrarlanan veri oluşabilir.
-Manuel seçilen datalar yine raw images de kullanılan formatta "dataset" dosya yoluna kopyalanır. Sonrasında "data_splitter.py" çalıştırılır. Bu kod verileri %80-20 olarak şekilde böler ve train dosya formatında "model_dataset" dizinine kaydeder.


## Görüntüler
----------------------
![Ekran görüntüsü 2024-06-19 182220](https://github.com/baloglu321/Animal-classification-with-gradio/assets/98214109/b7c22038-35b4-45df-85a5-245348581623)
![Ekran görüntüsü 2024-06-19 183127](https://github.com/baloglu321/Animal-classification-with-gradio/assets/98214109/44697ebb-bd7c-4985-b9cb-254ee67aeead)




