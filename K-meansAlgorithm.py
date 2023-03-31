

'''Python içine gerekli kütüphaneler import edilir.Diğer kütüphaneler ilgili işlem yapılmadan önce aşağıda import edilecektir. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

'''Kullanacağımız veri seti Iris veri setidir.Python'ın pandas kütüphanesinin read_excel methodu ile veri setini import ediyoruz.'''

data = pd.read_excel("Iris.xls")

print(data) 

'''Veri setinde yer alan kolonlarda bağımlı değşiken kolonuna ait belirli özellikler bulunmaktadır.Bu özelliklerden birkaçını kullanarak makine öğrenmesinde gözetimsiz öğrenme 
yöntemlerinden biri olan K-means algoritması ile çalışacağız.Özelliklerimizden bir matris oluşturup v değişkenine atadık.'''

v=data.iloc[:,1:4].values

'''Bu çalışmada uygulayacağımız algoritma K-Means algoritmasıdır.K-means algoritmaları,makine öğrenmesi yöntemlerinden biri olup gözetimsiz öğrenme metodlarından biridir.Gözetimsiz
öğrenme ile kastedilen makineye önceden herhangi bir bilgi verilmeden makinenin var olan değişkenler arasındaki ilişkiyi öğrenerek kendi kendine tahmin yapmasına dayanır.Gözetimsiz 
öğrenmede yaygın olarak kullanılan K-means algoritmaları benzer niteliklere sahip olan verileri aynı kümeye sokar.Bu sayede var olan verileri belirli sayıda küme içerisine dahil eder.
Algoritmanın çalışma prensibi şöyledir:
1)Başlangıçta kaç küme olacağı parametre olarak belirlenir.
2)Her veri noktası en yakın küme merkezine atanır.
3)Her kümenin merkezi yeniden hesaplanır.
4)Küme merkezleri sabitlenene kadar bu adımlar devam eder.
  Çalışmaya K-means Scikit-Learn kütüphanesinden import edilerek başlanır.  '''
  
from sklearn.cluster import KMeans


'''K-Means algoritmasının dezavantajlarından birisi küme sayısının rastgele seçilebiliyor olmasıdır.Başlangıçta küme sayısının optimal seçilmesi yapılacak segmentlemenin de başarısını 
etkileyecektir.Optimal bir k değeri belirlemek için dirsek yöntemi kullanılır.Dirsek yöntemi adı verilen bu yöntemde WCSS değerine göre grafik yardımıyla optimal bir k değeri seçilmeye 
çalışılır. WCSS (Within Cluster Sum of Squares) değeri ise, k-means algoritması tarafından oluşturulan kümeleme sonuçlarının kalitesini ölçen bir metriktir.WCSS, her bir kümenin merkez
 noktası ile kümedeki tüm noktalar arasındaki mesafelerin kareleri toplamından oluşur. Her bir kümenin WCSS değeri hesaplanarak, tüm kümeleme için toplam WCSS değeri hesaplanır.
 İdeal olan, kümeleme sonucunda WCSS değeri mümkün olduğunca düşük olmasıdır. Bu, her bir kümenin mümkün olduğunca homojen olduğunu ve kümeleme sonucunun kaliteli olduğunu gösterir.
 Aşağıda wcss adında bir liste oluşturup 1-11 arasında bir küme sayısı oluşturuyoruz.Küme sayısı birer birer artacak ve her bir küme sayısının da WCSS değeri hesaplanarak wcss listesine
 eklenecektir.K-means algoritmasını kullanırken belli parametreleri de aşağıda verdik.
-- n_cluster=küme sayısı
-- init=Başlangıç merkez noktalarını belirleyen parametredir.Parametre belirlenmediyse default olarak (k-means++) kullanılır.'''

wcss = []
kume_sayisi=range(1, 11)
for i in kume_sayisi:
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(v)
    wcss.append(kmeans.inertia_)
    
'''Oluşturulan WCSS grafiğinin matplotlib kütüphanesi yardımıyla görselleştirilmesi sağlanır.Grafikte optimal değerin belirlenmesiyle n_cluster parametresi yerine yazılır.'''    
 
plt.plot(kume_sayisi, wcss)
plt.title('Küme Sayısı Belirlemek için Dirsek Yöntemi')
plt.xlabel('Küme Sayısı')
plt.ylabel('WCSS')
plt.show()

'''K-means kütüphanesini import ettik,WCSS grafiği yardımıyla optimal k değerine ulaştıktan sonra km adında bir nesne oluşturduk.Oluşturulan nesnenin belirli parametreleri alacağından
yukarıda bahsedildi.n_cluster yani küme sayımız için optimal değer 4 olarak yorumlandı.İnit parametresi için ise k-means++ seçildi.K-means++ algoritması, küme merkezlerinin
 daha iyi bir şekilde seçilmesini sağlamak için öncelikle ilk küme merkezini rastgele seçer ve sonraki küme merkezlerinin seçiminde uzaklıklara dayalı bir seçim yapar.
 Sonrasında fit methodu ile makine eğitilip predict methodu ile makine tahminlemede bulundu.'''
 
km = KMeans(n_clusters=4, init='k-means++',random_state=0)   
km.fit(v)
predict=km.predict(v)


'''Makinenin tahminlerinin ayrı ayrı grafikte gösterimi için aşağıda saçılım grafiğine yer verilmiştir.'''
import matplotlib.pyplot as plt
plt.scatter(v[predict==0,0],v[predict==0,1],s=50,color='red')
plt.scatter(v[predict==1,0],v[predict==1,1],s=50,color='blue')
plt.scatter(v[predict==2,0],v[predict==2,1],s=50,color='green')
plt.scatter(v[predict==3,0],v[predict==3,1],s=50,color='yellow')
plt.title('K-Means Iris Dataset')
plt.show()


