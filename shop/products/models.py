from django.db import models

from shop.categories.models import Categories


class Products(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=30)
    price = models.IntegerField()
    image_url = models.CharField(max_length=255)

    category_id = models.ForeignKey(Categories, on_delete=models.CASCADE)

    class Meta:
        db_table = "shop_products"

    def __str__(self):
        return f'{self.pk} {self.id} {self.name} {self.price} {self.image_url}'
