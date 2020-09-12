from django.db import models
from django.urls import reverse

# Create your models here.
class Network(models.Model):
	n_reticulations = models.PositiveSmallIntegerField(default=0)
	n_taxa = models.PositiveSmallIntegerField(default=0)
	speciation_rate = models.FloatField(default=20.0)
	hybridization_rate = models.FloatField(default=10.0)
	time_limit =  models.FloatField(default=0.1)
	sites_per_tree = models.PositiveSmallIntegerField(default=100)
	newick_path = models.TextField(default=None, null=True, blank=True)
	msa_path = models.TextField(default=None, null=True, blank=True)
	image_path = models.TextField(default=None, null=True, blank=True)

	def get_absolute_url(self):
		return reverse("network:network-detail", kwargs={"pk": self.pk})