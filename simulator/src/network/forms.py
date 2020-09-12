from django import forms

from .models import Network

class NetworkModelForm(forms.ModelForm):
	class Meta:
		model = Network
		fields = [
			'n_reticulations',
			'n_taxa',
			'speciation_rate',
			'hybridization_rate',
			'time_limit',
			'sites_per_tree',
		]
		#widgets = {
		#	'sites_per_tree' : forms.NumberInput(attrs={'type': 'range', 'step': '1', 'min': '1', 'max': '10000'}),
		#}