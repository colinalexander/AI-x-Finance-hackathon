// #let main_font = "D-DIN"
#let main_font = "Arial"
#set page(margin: (left: 0.5in, right: 0.5in, top: 0.5in, bottom: 0.5in))

//substitute

// Ticker symbol at top of page
#set text(font: main_font, size: 50pt)
#text("$" + ticker_symbol)

// Recommendation text

#align(bottom)[
  #set text(font: main_font, size: 30pt)
  #text("Recommendation:")

  #set text(font: main_font, size: 50pt)
  #v(-45pt)

  #{
    if recommendation == -2 {
      text("Strong Sell")
    } else if recommendation == -1 {
      text("Sell")
    } else if recommendation == 0 {
      text("Hold")
    } else if recommendation == 1 {
      text("Buy")
    } else if recommendation == 2 {
      text("Strong Buy")
    }
  }
]

// Bring rec text down
#v(-50pt)

// Arrow
#align(center)[
  #align(bottom)[
    #h(200pt * recommendation)
    #text("⏷")
  ]
]

// Bring arrow closer to stuff
#v(-40pt)

// Graphic at bottom of page
#set text(font: main_font, size: 30pt)
#align(center)[
  #align(bottom)[
    #text("📉📉")
    #h(1fr)
    #text("📉")
    #h(1fr)
    #text("⏹️")
    #h(1fr)
    #text("📈")
    #h(1fr)
    #text("📈📈")
  ]
]