from django import forms

from .models import Network

class NetworkModelForm(forms.ModelForm):
	class Meta:
		model = Network
		fields = [
			'output_base',
			'incomplete_lineage_sorting',
			'population_size',
			'speciation_rate',
			'hybridization_rate',
			'time_limit',
			'n_trees',
			'sites_per_tree',
			'benchmark_mode',
		]
		#widgets = {
		#	'sites_per_tree' : forms.NumberInput(attrs={'type': 'range', 'step': '1', 'min': '1', 'max': '10000'}),
		#}