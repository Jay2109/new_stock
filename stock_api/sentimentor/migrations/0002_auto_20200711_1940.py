# Generated by Django 3.0.8 on 2020-07-11 19:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sentimentor', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tickersentiment',
            name='sentiment',
            field=models.CharField(max_length=20),
        ),
        migrations.AlterField(
            model_name='tickersentiment',
            name='sym_name',
            field=models.CharField(max_length=20),
        ),
    ]
