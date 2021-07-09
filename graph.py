
from py2neo import Graph
model = Graph

graph=Graph(uri="bolt://localhost:7687",auth=("neo4j","1234"))
host = "bolt://localhost:7687"
user = 'neo4j'
password = '1234'


from IPython.display import IFrame, HTML
import json
import uuid


def generate_vis(host, user, password, cypher, labels_json, relationships_json):
    html = """<html>
<head>
    <title>Neovis.js Simple Example</title>
            <style type="text/css">
                html, body {{
                    font: 16pt arial;
                }}
                #viz {{
                    width: 400px;
                    height: 450px;
                    font: 22pt arial;
                }}
            </style>
            <script src="https://cdn.neo4jlabs.com/neovis.js/v1.1.0/neovis.js"></script>
            <script
                    src="https://code.jquery.com/jquery-3.2.1.min.js"
                    integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4="
                    crossorigin="anonymous"></script>
            <script type="text/javascript">
                var viz;
                function draw() {{
                    var config = {{
                        container_id: "viz",
                        server_url: "{host}",
                        server_user: "{user}",
                        server_password: "{password}",
                        labels: {labels},
                        relationships: {relationships},
                        initial_cypher: "{cypher}"
                    }};
                    viz = new NeoVis.default(config);
                    viz.render();                    
                    viz.onVisualizationRendered(function(ctx) {{
                        let imageSrc = document.getElementsByTagName("canvas")[0].toDataURL();
                        console.log(imageSrc);
                        document.getElementById("viz-image").src=imageSrc;
                        //document.getElementById("viz").style="display:none";
                        
                        let kernel = IPython.notebook.kernel;
                        //let command = 'display(HTML('<img id="viz-image" width="300px" src="' + imageSrc + '" />';
                        let command = "foo = 'bar'";
                        kernel.execute(command);
                        
                    }});
                }}
            </script>
         </head>
        <body onload="draw()">
            <div id="viz"></div>
        </body>
    </html>
    """

    html = html.format(
        host=host,
        user=user,
        password=password,
        cypher=cypher,
        labels = json.dumps(labels_json),
        relationships=json.dumps(relationships_json)
    )

    unique_id = str(uuid.uuid4())
    filename = "graph-{}.html".format(unique_id)

    with open(filename, "w") as f:
        f.write(html)
    return IFrame(src=filename, width=500, height=500)



cypher = " MATCH (p)-[:MAKES_PRODUCT]-(m:Product)-[:IMPORTED_FROM]-(c:Country) WHERE m.sector='healthcare' AND m.type='Masks' AND c.name='China' RETURN * "
labels_json = {
    "Person": {
        "caption": "name",
        "size": "pagerank",
        "community": "community"
    }
}

relationships_json = {
    "RELATED": {
        "thickness": "id",
        "caption": False
    }
}

generate_vis(host, user, password, cypher, labels_json, relationships_json)




