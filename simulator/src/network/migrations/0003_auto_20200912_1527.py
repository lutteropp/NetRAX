# Generated by Django 3.1.1 on 2020-09-12 15:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('network', '0002_auto_20200912_0917'),
    ]

    operations = [
        migrations.AlterField(
            model_name='network',
            name='image_path',
            field=models.TextField(default=None),
        ),
        migrations.AlterField(
            model_name='network',
            name='msa_path',
            field=models.TextField(default=None),
        ),
        migrations.AlterField(
            model_name='network',
            name='newick_path',
            field=models.TextField(default=None),
        ),
    ]
