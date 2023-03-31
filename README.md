# Machine-Learning-K-Means-Clustering
# K-Means Clustering
Bu kod Python'da yazılmış olup, makine öğrenmesi ve veri analizi için gerekli kütüphaneleri, pandas, numpy ve matplotlib gibi, içe aktarır. Makine öğrenmesi topluluğunda iyi bilinen Iris veri kümesini kullanır.

Kod, Iris veri kümesinin bazı özelliklerini içeren bir matris oluşturur ve benzer veri noktalarını gruplandırmak için denetimsiz bir öğrenme yöntemi olan K-ortalama algoritmasını uygular. K-ortalama algoritması, öncelikle belirli bir sayıda küme (k) rastgele seçerek ve ardından yakınsama elde edilene kadar veri noktalarını en yakın küme merkezine atayarak çalışır.

Optimal küme sayısını belirlemek için kod, her küme sayısı için küme içi kare toplamlarını (WCSS) hesaplayarak ve bunları küme sayısıyla karşılaştırarak "dirsek yöntemi"ni kullanır. Optimal küme sayısı, WCSS'nin düzleşmeye başladığı "dirsek" noktasıdır.

Genel olarak, bu kod, K-ortalama algoritmasını benzer veri noktalarını gruplandırmak için nasıl kullanacağını ve dirsek yöntemini kullanarak optimal küme sayısını nasıl belirleyeceğini göstermektedir.
