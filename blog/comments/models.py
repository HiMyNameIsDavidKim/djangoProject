from django.db import models

from blog.busers.models import BUsers
from blog.posts.models import Posts


class Comments(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now_add=True)
    parent_id = models.TextField(null=True)

    user_id = models.ForeignKey(BUsers, on_delete=models.CASCADE)
    post_id = models.ForeignKey(Posts, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_comments"

    def __str__(self):
        return f'{self.pk} {self.id} {self.content} {self.created_at} {self.updated_at} {self.parent_id}'
