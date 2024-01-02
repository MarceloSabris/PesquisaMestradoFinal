SELECT  passo,acao,
	CONCAT ('[', ROUND( CAST( AVG( acuracy_questao_0 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_1 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_2 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_3 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_4 ) as NUMERIC), 5 ) ,']') from "Curriculos1"
where porcentagem = '0.5,0.5,0.5,0.1,0.1' and  tipoescolha ='manual'
GROUP BY passo,acao order by passo


SELECT count(*), passo,acao  FROM "Curriculos1" where curriculo 
like 'ExecucaoAutomatica_19122023_1628_exec_%%' 
GROUP BY passo, acao ORDER BY "passo" LIMIT 100

--ExecucaoAutomatica_18122023_1218_exec_ -> 3 
--ExecucaoAutomatica_19122023_0121_exec_% => 4
--ExecucaoAutomatica_19122023_1628_exec_% => 5 


select * from "Curriculos1" where curriculo like '%toma%' ORDER BY "id" asc LIMIT 100 OFFSET 200 
-- pagina 3