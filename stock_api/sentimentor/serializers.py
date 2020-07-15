from rest_framework import serializers
from sentimentor.models import Sentimental,Tickersentiment


class Sentimentorserializer(serializers.ModelSerializer):
    class Meta:
        model=Sentimental
        fields="__all__"



class Tickerserializer(serializers.ModelSerializer):
    class Meta:
        model=Tickersentiment
        fields="__all__"
