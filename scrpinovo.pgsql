

SELECT  passo,acao,
	CONCAT ('[', ROUND( CAST( AVG( acuracy_questao_0 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_1 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_2 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_3 ) as NUMERIC), 5 ) ,',',
                 ROUND( CAST( AVG( acuracy_questao_4 ) as NUMERIC), 5 ) ,']') from "Curriculos1"
where curriculo = 'episodioAleatorio1' 
GROUP BY passo,acao order by passo


select * from "Curriculos1" where curriculo = 'ExecucaoAleatoria_0_27122023_1915_exec_0' 

    update "Curriculos1" set curriculo = 'episodioAleatorio3' where curriculo = 'ExecucaoAleatoria_0_29122023_1204_exec_0'

SELECT curriculo, count(*) FROM "Curriculos1" where curriculo like '%Alea%' group by curriculo LIMIT 100

update "Curriculos1" set curriculo = 'episodioAleatorio3' where curriculo = 'ExecucaoAleatoria_0_29122023_1204_exec_0'
