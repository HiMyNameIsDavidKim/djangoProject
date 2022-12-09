from django.db import models

from shop.products.models import Products
from shop.susers.models import SUsers


class Carts(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)

    product_id = models.ForeignKey(Products, on_delete=models.CASCADE)
    user_id = models.ForeignKey(SUsers, on_delete=models.CASCADE)


    class Meta:
        db_table = "shop_carts"

    def __str__(self):
        return f'{self.pk} {self.id}'
