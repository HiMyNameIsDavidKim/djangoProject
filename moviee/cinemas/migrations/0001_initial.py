# Generated by Django 4.1 on 2022-11-30 07:07

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Cinemas',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=20)),
                ('image_url', models.TextField()),
                ('address', models.CharField(max_length=50)),
                ('detail_address', models.CharField(max_length=30)),
            ],
            options={
                'db_table': 'movie_cinemas',
            },
        ),
    ]
