# Generated by Django 4.1 on 2022-11-30 07:07

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BUsers',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('email', models.CharField(max_length=120)),
                ('nickname', models.CharField(max_length=20)),
                ('password', models.CharField(max_length=255)),
            ],
            options={
                'db_table': 'blog_users',
            },
        ),
    ]
