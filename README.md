# Supernova Identification

Source code and processed data from the Article 

For practical reasons, different steps of the data frames used on this project were saved on pickle files. 

Those pickles will be available for download at the following link: https://drive.google.com/drive/folders/1FP3515CzNqQJY0PZFSWT9y7gC8ldi8pm?usp=sharing as well as the raw data.

They must be unzipped inside a <data/> folder on the root of this project. Inside the <data/> folder there will be 2 following folders: <data/raw_data/> containing all the .txt raw files and the <data/processed/> containing all the pickles used on the notebooks during the different parts of the project.

#### Notes:
https://arxiv.org/pdf/1806.06607.pdf
https://arxiv.org/pdf/2009.12112.pdf

	- Sigma MAD:
		-> Sigma MAD, desvio quadratico médio da mediana https://arxiv.org/pdf/1806.06607.pdf (4.1. Metrics)

	- Selecionar só IA
		DONE -> (tirar os outliers obviamente, que sao os zuados, ver pelos boxplots)
			R: Só piora
		DONE -> Tirar os outliers de antes, aqueles que tem poucos pontos de interpolação.
			R: Melhor resposta foi quando o limiar foi 4 pontos e mesmo assim só foi 8^10-4 de melhora, muito insignificante e sem nem considerar a valiação em cima daqueles que ele não treinou.
		-> Menor do que 0.1 ta bem razoavel. 0.03*(1 + Z)

	- Ver se tem como abrir as árvores no XGBoost
		-> Ver o valor de redshift que cada Decision tree deu e plotar o histograma (o objetivo final) ter a pdf do redshift		

	- Outros Detalhes:
		-> Analisar tempo de processamento e tempo gasto durante pipeline (ver como eu faço as magics,tem que ser por linha provavelmente)
			. DONE Não funcionou, demorou 4-5 horas um GP Buscar vetorizar com numpy array o GP .
		-> Analisar melhor os describes e as distribuições reais e as previstas do melhor algoritmo
		-> NAO SEI SE VALE Rodar a regressão com os valores do GP (antes mesmo dos wavelets)
		-> Auto ML: h2o.AI, autoKeras
