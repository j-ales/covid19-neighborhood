# modified from http://cssmenumaker.com/br/blog/stylish-css-tables-tutorial
css = """
<style type=\"text/css\">
table {
color: #7533;
font-family: Helvetica, Arial, sans-serif;
width: 1200px;
border-collapse:
collapse; 
border-spacing: 0;
}
td, th {
border: 1px solid transparent; /* No more visible border */
height: 30px;
}
th {
background: #d64b51; /* Darken header a bit */
font-weight: bold;
text-align: center;
}
td {
background: #ffbfc2;
text-align: center;
}
table tr:nth-child(odd) td{
background-color: white;
}
</style>
"""