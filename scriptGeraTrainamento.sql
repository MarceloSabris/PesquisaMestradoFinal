SELECT  passo,acao,
	CONCAT ('[', ROUND( CAST( AVG( acuracy_questao_0 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_1 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_2 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_3 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_4 ) as NUMERIC), 5 ) ,']') from "Curriculos1"
where curriculo like 'ExecucaoManual_1,1,1,1,1_exec_%'
GROUP BY passo,acao order by passo
