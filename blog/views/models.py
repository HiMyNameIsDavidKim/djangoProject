from django.db import models

from blog.busers.models import BUsers
from blog.posts.models import Posts


class Views(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    ip_address = models.DateTimeField(auto_now=True)

    user_id = models.ForeignKey(BUsers, on_delete=models.CASCADE)
    post_id = models.ForeignKey(Posts, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_views"

    def __str__(self):
        return f'{self.pk} {self.id} {self.ip_address}'
