// Copyright 2019 The Route Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"math"
	"math/cmplx"
	"math/rand"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tc128"
)

const (
	// Width of data
	Width = 4
	// Middle is the size of the middle layer
	Middle = 3
	// Eta is the learning rate
	Eta = .6
)

// Pair is a pair of training dtaa
type Pair struct {
	iris.Iris
	Input, Output []complex128
}

// Factorial computes the factorial of a number
func Factorial(n int) int {
	if n > 0 {
		return n * Factorial(n-1)
	}
	return 1
}

func printTable(out io.Writer, headers []string, rows [][]string) {
	sizes := make([]int, len(headers))
	for i, header := range headers {
		sizes[i] = len(header)
	}
	for _, row := range rows {
		for j, item := range row {
			if length := len(item); length > sizes[j] {
				sizes[j] = length
			}
		}
	}

	last := len(headers) - 1
	fmt.Fprintf(out, "| ")
	for i, header := range headers {
		fmt.Fprintf(out, "%s", header)
		spaces := sizes[i] - len(header)
		for spaces > 0 {
			fmt.Fprintf(out, " ")
			spaces--
		}
		fmt.Fprintf(out, " |")
		if i < last {
			fmt.Fprintf(out, " ")
		}
	}
	fmt.Fprintf(out, "\n| ")
	for i, header := range headers {
		dashes := len(header)
		if sizes[i] > dashes {
			dashes = sizes[i]
		}
		for dashes > 0 {
			fmt.Fprintf(out, "-")
			dashes--
		}
		fmt.Fprintf(out, " |")
		if i < last {
			fmt.Fprintf(out, " ")
		}
	}
	fmt.Fprintf(out, "\n")
	for _, row := range rows {
		fmt.Fprintf(out, "| ")
		last := len(row) - 1
		for i, entry := range row {
			spaces := sizes[i] - len(entry)
			fmt.Fprintf(out, "%s", entry)
			for spaces > 0 {
				fmt.Fprintf(out, " ")
				spaces--
			}
			fmt.Fprintf(out, " |")
			if i < last {
				fmt.Fprintf(out, " ")
			}
		}
		fmt.Fprintf(out, "\n")
	}
}

func main() {
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	items, n := datum.Fisher, Factorial(Width)
	length := n * len(items)
	pairs := make([]Pair, 0, length)
	for _, item := range items {
		permutations := make([][]complex128, 0, n)
		c, i, a := make([]int, Width), 0, make([]complex128, Width)
		for j, measure := range item.Measures {
			a[j] = cmplx.Rect(measure, float64(j)*math.Pi/2)
		}

		permutation := make([]complex128, Width)
		copy(permutation, a)
		permutations = append(permutations, permutation)

		for i < Width {
			if c[i] < i {
				if i&1 == 0 {
					a[0], a[i] = a[i], a[0]
				} else {
					a[c[i]], a[i] = a[i], a[c[i]]
				}

				permutation := make([]complex128, Width)
				copy(permutation, a)
				permutations = append(permutations, permutation)

				c[i]++
				i = 0
				continue
			}
			c[i] = 0
			i++
		}

		output := permutations[0]
		for _, input := range permutations {
			pair := Pair{
				Iris:   item,
				Input:  input,
				Output: output,
			}
			pairs = append(pairs, pair)
		}
	}
	if len(pairs) != length {
		panic("invalid length")
	}
	fmt.Println("pairs", len(pairs))

	rnd := rand.New(rand.NewSource(1))
	random128 := func(a, b float64) complex128 {
		return complex((b-a)*rnd.Float64()+a, (b-a)*rnd.Float64()+a)
	}

	parameters := make([]*tc128.V, 0, 4)
	w0, b0 := tc128.NewV(Width, Middle), tc128.NewV(Middle)
	w1, b1 := tc128.NewV(Middle, Width), tc128.NewV(Width)
	parameters = append(parameters, &w0, &b0, &w1, &b1)
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, random128(-1, 1))
		}
	}

	input, output := tc128.NewV(Width, length), tc128.NewV(Width, length)
	l0 := tc128.Sigmoid(tc128.Add(tc128.Mul(w0.Meta(), input.Meta()), b0.Meta()))
	l1 := tc128.Add(tc128.Mul(w1.Meta(), l0), b1.Meta())
	cost := tc128.Avg(tc128.Quadratic(l1, output.Meta()))

	inputs, outputs := make([]complex128, 0, Width*length), make([]complex128, 0, Width*length)
	for _, pair := range pairs {
		inputs = append(inputs, pair.Input...)
		outputs = append(outputs, pair.Output...)
	}
	input.Set(inputs)
	output.Set(outputs)

	iterations := 256
	pointsAbs, pointsPhase := make(plotter.XYs, 0, iterations), make(plotter.XYs, 0, iterations)
	for i := 0; i < iterations; i++ {
		for _, p := range parameters {
			p.Zero()
		}
		input.Zero()
		output.Zero()

		total := tc128.Gradient(cost).X[0]

		norm := float64(0)
		for _, p := range parameters {
			for _, d := range p.D {
				norm += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm = math.Sqrt(norm)
		if norm > 1 {
			scaling := 1 / norm
			for _, p := range parameters {
				for l, d := range p.D {
					p.X[l] -= Eta * d * complex(scaling, 0)
				}
			}
		} else {
			for _, p := range parameters {
				for l, d := range p.D {
					p.X[l] -= Eta * d
				}
			}
		}

		pointsAbs = append(pointsAbs, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		pointsPhase = append(pointsPhase, plotter.XY{X: float64(i), Y: float64(cmplx.Phase(total))})
	}

	plot := func(title, name string, points plotter.XYs) {
		p, err := plot.New()
		if err != nil {
			panic(err)
		}

		p.Title.Text = title
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, name)
		if err != nil {
			panic(err)
		}
	}
	plot("cost abs vs epochs", "cost_abs.png", pointsAbs)
	plot("cost phase vs epochs", "cost_phase.png", pointsPhase)

	readme, err := os.Create("README.md")
	if err != nil {
		panic(err)
	}
	defer readme.Close()

	{
		input := tc128.NewV(Width)
		l0 := tc128.Sigmoid(tc128.Add(tc128.Mul(w0.Meta(), input.Meta()), b0.Meta()))

		headers, rows := make([]string, 0, 1+2*Middle), make([][]string, 0, length)
		headers = append(headers, "label")
		for i := 0; i < Middle; i++ {
			headers = append(headers, fmt.Sprintf("abs %d", i))
			headers = append(headers, fmt.Sprintf("phase %d", i))
		}
		for _, item := range items {
			inputs := make([]complex128, Width)
			for j, measure := range item.Measures {
				inputs[j] = cmplx.Rect(measure, float64(j)*math.Pi/2)
			}
			input.Set(inputs)
			row := make([]string, 0, 1+2*Middle)
			l0(func(a *tc128.V) bool {
				row = append(row, item.Label)
				for _, value := range a.X {
					row = append(row, fmt.Sprintf("%f", cmplx.Abs(value)))
					row = append(row, fmt.Sprintf("%f", cmplx.Phase(value)))
				}
				return true
			})
			rows = append(rows, row)
		}
		printTable(readme, headers, rows)
	}
}
