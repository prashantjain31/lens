package com.example.lens;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.lens.ml.MobilenetV110224Quant;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class MainActivity extends AppCompatActivity {
    //Required Variables
    Bitmap bitmap = null;
    TextView textView;
    ImageView imageView;
    HashMap<Integer, String> wordList = new HashMap<>();

    // onCreate method
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Opens file and convert it into array
        String fileName = "labels.txt";
        AssetManager manager = getAssets();
        try {
            InputStream inputStream = manager.open(fileName);
            InputStreamReader inputStreamReader = new InputStreamReader(inputStream);

            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            String line = bufferedReader.readLine();
            int i = 0;
            while (!TextUtils.isEmpty(line)) {
                wordList.put(i, line);
                line = bufferedReader.readLine();
                i++;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        imageView = (ImageView) findViewById(R.id.imageView);
        textView = (TextView) findViewById(R.id.textView);
        Button select = (Button) findViewById(R.id.button);
        Button predict = (Button) findViewById(R.id.button2);

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent getIntent = new Intent(Intent.ACTION_GET_CONTENT);
                getIntent.setType("image/*");

                Intent pickIntent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                pickIntent.setType("image/*");

                Intent chooserIntent = Intent.createChooser(getIntent, "Select Image");
                chooserIntent.putExtra(Intent.EXTRA_INITIAL_INTENTS, new Intent[] {pickIntent});

                startActivityForResult(chooserIntent, 100);
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
                TensorImage tbuffer = TensorImage.fromBitmap(resized);
                ByteBuffer byteBuffer = tbuffer.getBuffer();
                try {
                    MobilenetV110224Quant model = MobilenetV110224Quant.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    MobilenetV110224Quant.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    float[] predictionArray = outputFeature0.getFloatArray();
                    textView.setText(wordList.get(getMax(predictionArray)));

                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        imageView.setImageURI(data.getData());
        Uri uri = data.getData();
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    int getMax(float[] arr) {
        int ind = 0;
        float min = 0.0f;
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] > min) {
                ind = i;
                min = arr[i];
            }
        }
        return ind;
    }

}