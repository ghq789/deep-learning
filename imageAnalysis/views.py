from django.shortcuts import render
from rest_framework.views import APIView


# Create your views here.
class Analysis(APIView):
    def get(self, request):
        # 得到图片数据
        data = request.data['image']
        #
        return render(request, 'imageAnalysis/analysis.html')