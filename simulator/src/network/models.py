from django.db import models
from django.urls import reverse

# Create your models here.
class Network(models.Model):
	output_base = models.CharField(max_length=200, default=None, null=False, blank=False)
	n_reticulations = models.PositiveSmallIntegerField(default=0, null=True, blank=True)
	n_taxa = models.PositiveSmallIntegerField(default=0, null=True, blank=True)
	speciation_rate = models.FloatField(default=20.0)
	hybridization_rate = models.FloatField(default=10.0)
	population_size = models.FloatField(default=0.01)
	time_limit =  models.FloatField(default=0.1)
	n_trees = models.PositiveSmallIntegerField(default=1)
	sites_per_tree = models.PositiveSmallIntegerField(default=100)
	newick_path = models.TextField(default=None, null=True, blank=True)
	newick_dendroscope_path = models.TextField(default=None, null=True, blank=True)
	msa_path = models.TextField(default=None, null=True, blank=True)
	image_path = models.TextField(default=None, null=True, blank=True)
	incomplete_lineage_sorting = models.BooleanField(default=False, null=False, blank=True)

	def get_absolute_url(self):
		return reverse("network:network-detail", kwargs={"pk": self.pk})